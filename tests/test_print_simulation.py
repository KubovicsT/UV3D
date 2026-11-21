"""Tests for the interactive offline print simulation and its display-only geometry."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from uv3d.simulation_engine import (
    PrintSimulation, format_sim_time, TIMELINE_TICKS)
from uv3d.simulation_widget import (
    SimulationWidget, read_layer_textures)
from uv3d.support_surface import (
    build_support_surface, merge_rectangles, voxel_surface)
from uv3d.simulation_viewport import (
    make_head_cuboid, SIM_LAYER_FRAGMENT)


def make_job(folder, layers=3, support_enabled=True):
    folder.mkdir(parents=True, exist_ok=True)
    w, h = 50, 20
    records = []
    for idx in range(layers):
        basename = f"{idx+1:05d}"
        model = np.zeros((h,w), dtype=np.uint8)
        model[7:14,14:28] = 255
        sacrificial = np.zeros((h,w), dtype=np.uint8)
        sacrificial[5:16,5:14] = 255
        painted = np.zeros((h,w,4), dtype=np.uint8)
        painted[7:14,14:28] = [45,155,235,255]  # BGRA -> RGBA conversion
        for key,data in (("white",model),("varnish",np.zeros_like(model)),
                         ("color",painted)):
            assert cv2.imencode(".png",data)[0]
            (folder/f"{basename}_{key}.png").write_bytes(
                cv2.imencode(".png",data)[1].tobytes())
        record = {
            "index":idx+1,
            "slice_center_z_mm":.1+idx*.2,
            "target_stack_height_mm":(idx+1)*.2,
            "white":f"{basename}_white.png",
            "varnish":f"{basename}_varnish.png",
            "color":f"{basename}_color.png",
        }
        if support_enabled:
            name=f"{basename}_support.png"
            (folder/name).write_bytes(cv2.imencode(".png",sacrificial)[1].tobytes())
            record["support"]=name
        records.append(record)
    settings={"dpi":254,"layer_mm":.2,"margin_mm":0,
              "body_material":"white","support_enabled":support_enabled}
    manifest={
        "format":"UV3D_OFFLINE_JOB_V1","source_name":"example.ply",
        "settings":settings,"effective_raster_margin_mm":0,
        "raster_size_px":[w,h],
        "model_bounds_mm":[[0,-2,0],[5,0,layers*.2]],
        "scaled_dimensions_mm":[5,2,layers*.2],
        "support":{"enabled":support_enabled},
        "layers":records,
    }
    job=folder/"job.json"
    job.write_text(json.dumps(manifest),encoding="utf-8")
    return job,manifest


def test_serpentine_head_moves_and_current_layer_progress_is_seekable(tmp_path):
    _,manifest=make_job(tmp_path/"job")
    plan=PrintSimulation(manifest,move_speed_mm_s=10,
                         scan_pitch_mm=1,head_clearance_mm=2)
    assert plan.rows==2
    assert plan.path_length_mm==11
    start=plan.get_frame(0)
    assert start.layer_index==0
    assert start.layer_progress==pytest.approx(0)
    assert start.head_position_mm[:2]==pytest.approx((0,-.5))
    assert start.head_position_mm[2]==pytest.approx(.2+2)
    assert plan.total_seconds==pytest.approx(3*(11/10+.25))
    quarter=plan.get_frame(.25)
    assert quarter.layer_index==0
    assert quarter.layer_progress>.0
    midpoint=plan.get_frame(.5)
    assert midpoint.layer_index==1
    assert midpoint.completed_height_mm==pytest.approx(.2)
    final=plan.get_frame(1)
    assert final.layer_index==2
    assert final.completed
    assert final.current_top_mm==pytest.approx(.6)
    assert final.completed_height_mm==pytest.approx(.6)
    assert final.layer_progress==pytest.approx(1)
    assert plan.position_from_ticks(TIMELINE_TICKS//2).layer_index==1


def test_speed_changes_duration_and_layer_step(tmp_path):
    _,job=make_job(tmp_path/"job",layers=4)
    plan=PrintSimulation(job,move_speed_mm_s=20)
    initial=plan.total_seconds
    plan.set_move_speed(80)
    assert plan.total_seconds<initial
    assert plan.get_frame(.5).layer_index==2
    assert plan.step_layer(.25,1)==pytest.approx(.5)
    assert format_sim_time(3661)=="01:01:01"
    for speed in (0,-100,float("nan"),1000):
        with pytest.raises(ValueError):
            plan.set_move_speed(speed)


def test_mask_loading_preserves_color_and_support_without_overwriting_files(tmp_path):
    job,_=make_job(tmp_path/"job")
    body,color,support=read_layer_textures(job,1,max_side=1024)
    assert body.shape==(20,50)
    assert color.shape==(20,50,4)
    assert support.shape==(20,50)
    assert body[10,20]==255
    assert support[10,10]==255
    assert support[10,20]==0
    # The stored image is BGRA; the simulation loader converts it to RGBA.
    assert tuple(color[10,20])==(235,155,45,255)
    downscaled=read_layer_textures(job,1,max_side=25)
    assert downscaled[0].shape==(10,25)


def test_merged_rectangle_mesh_is_a_solid_surface_not_just_wireframe():
    grid=np.zeros((3,6,8),dtype=bool)
    grid[0:2,1:5,2:7]=True
    x=np.arange(9,dtype=np.float32)
    y=np.arange(7,dtype=np.float32)
    z=np.arange(4,dtype=np.float32)
    triangles=voxel_surface(grid,x,y,z)
    assert triangles.shape[1]==3
    assert len(triangles)>0 and len(triangles)%6==0
    assert triangles[:,0].min()==pytest.approx(2)
    assert triangles[:,0].max()==pytest.approx(7)
    assert triangles[:,2].max()==pytest.approx(2)
    rects=list(merge_rectangles(grid[0]))
    assert sum((b-a)*(d-c) for a,b,c,d in rects)==20


def test_support_surface_samples_real_mask_files(tmp_path):
    job,_=make_job(tmp_path/"job",layers=8)
    surface=build_support_surface(job,max_xy=48,max_levels=5)
    assert surface.triangles.shape[1]==3
    assert len(surface.triangles)>0
    assert surface.sampled_levels==5
    assert surface.triangles[:,2].min()>=0
    assert surface.triangles[:,2].max()<=1.6+1e-5


def test_3d_printhead_is_a_colored_cuboid_and_material_shader_has_alpha():
    head=make_head_cuboid(3,2,5,20,20)
    assert len(head)==36
    assert float(head["position"][:,2].min())==pytest.approx(5)
    assert len(np.unique(head["color"],axis=0))>=3
    assert "uSupportTint.a" in SIM_LAYER_FRAGMENT
    assert "uRowProgress" in SIM_LAYER_FRAGMENT
    assert "uColor" in SIM_LAYER_FRAGMENT


def test_simulation_widget_timeline_controls_without_gpu(tmp_path):
    app=QApplication.instance() or QApplication([])
    job,_=make_job(tmp_path/"job",layers=5)
    ui=SimulationWidget(tmp_path,gpu_enabled=False)
    ui.load_job(job)
    assert ui.plan.count==5
    assert ui.timeline.isEnabled()
    ui.timeline.setValue(60_000)
    assert ui.plan.get_frame(ui.elapsed/ui.plan.total_seconds).layer_index==3
    normalized=ui.elapsed/ui.plan.total_seconds
    ui.move_speed.setValue(200)
    assert ui.elapsed/ui.plan.total_seconds==pytest.approx(normalized)
    ui.support_opacity.setValue(72)
    assert ui.opacity_readout.text()=="72%"
    ui.toggle_play()
    assert ui.playing
    ui.pause()
    assert not ui.playing
    ui.step_layer(-1)
    assert ui.plan.get_frame(ui.elapsed/ui.plan.total_seconds).layer_index==2
    ui.close()
