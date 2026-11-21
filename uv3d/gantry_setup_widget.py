"""Gantry setup page inside the independent 3D simulation window.

Changes update the virtual route/ETA immediately and persist beneath
X:/UV3D/workspace/machine. Nothing is written to printer firmware.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QScrollArea, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from .gantry import (
    GantryConfig, HeadConfig, LampConfig, MATERIALS,
    default_gantry, gantry_file, load_gantry, save_gantry,
)


class GantrySetupWidget(QWidget):
    configurationChanged = Signal(object)

    HEAD_FIELDS = ("Name", "Material", "Offset X (mm)", "Offset Y (mm)",
                   "Body X (mm)", "Body Y (mm)", "Height (mm)",
                   "Swath (mm)", "Enabled")
    LAMP_FIELDS = ("Name", "Offset X (mm)", "Offset Y (mm)", "Size X (mm)",
                   "Size Y (mm)", "Height (mm)", "Enabled")

    def __init__(self, root: Path, parent=None):
        super().__init__(parent)
        self.root = Path(root)
        self.config = load_gantry(root)
        self._loading = False
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        layout = QVBoxLayout(body)
        intro = QLabel(
            "OFFLINE GANTRY LAYOUT — all positions are relative to the first "
            "printhead's center at X=0, Y=0. Every head and UV lamp moves "
            "together as one assembly. The starting layout contains "
            "UNMEASURED placeholders: calibrate these dimensions and offsets "
            "before relying on timing estimates. No printer commands are sent.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        head_box = QGroupBox("Printheads / material assignment")
        head_col = QVBoxLayout(head_box)
        self.head_table = QTableWidget(0, len(self.HEAD_FIELDS))
        self.head_table.setHorizontalHeaderLabels(self.HEAD_FIELDS)
        self.head_table.setMinimumHeight(230)
        self.head_table.setMinimumWidth(890)
        self.head_table.cellChanged.connect(self._commit)
        head_col.addWidget(self.head_table)
        head_buttons = QHBoxLayout()
        add_head = QPushButton("Add printhead")
        add_head.clicked.connect(self.add_head)
        remove_head = QPushButton("Remove selected printhead")
        remove_head.clicked.connect(self.remove_head)
        head_buttons.addWidget(add_head)
        head_buttons.addWidget(remove_head)
        head_col.addLayout(head_buttons)
        layout.addWidget(head_box)

        lamp_box = QGroupBox("UV lamps / curing footprint")
        lamp_col = QVBoxLayout(lamp_box)
        self.lamp_table = QTableWidget(0, len(self.LAMP_FIELDS))
        self.lamp_table.setHorizontalHeaderLabels(self.LAMP_FIELDS)
        self.lamp_table.setMinimumHeight(180)
        self.lamp_table.setMinimumWidth(720)
        self.lamp_table.cellChanged.connect(self._commit)
        lamp_col.addWidget(self.lamp_table)
        lamp_buttons = QHBoxLayout()
        add_lamp = QPushButton("Add UV lamp")
        add_lamp.clicked.connect(self.add_lamp)
        remove_lamp = QPushButton("Remove selected UV lamp")
        remove_lamp.clicked.connect(self.remove_lamp)
        lamp_buttons.addWidget(add_lamp)
        lamp_buttons.addWidget(remove_lamp)
        lamp_col.addLayout(lamp_buttons)
        layout.addWidget(lamp_box)

        cure_box = QGroupBox("Curing pass estimate")
        cure_layout = QFormLayout(cure_box)
        self.cure_passes = QSpinBox()
        self.cure_passes.setRange(1, 5)
        self.cure_passes.valueChanged.connect(self._commit)
        cure_layout.addRow("UV exposures per print pass", self.cure_passes)
        explanation = QLabel(
            "During left-to-right printing, only the lamp BEHIND the heads "
            "on the left shines. Right-to-left printing uses the right-side "
            "trailing lamp. The leading lamp stays off. If that lamp covers "
            "the printed band, UV curing happens during the printing scan "
            "without a separate sweep. After ink deposition, it continues "
            "until the lamp's BACK edge has completely cleared the model's "
            "far edge, maintaining the same speed as the printing scan. "
            "That gives each X position a full lamp-width exposure in the "
            "motion model. A non-trailing lamp makes an entire footprint "
            "sweep instead. Higher exposure counts add extra sweeps. "
            "UV power, optical intensity and real dose are NOT calibrated.")
        explanation.setWordWrap(True)
        cure_layout.addRow(explanation)
        layout.addWidget(cure_box)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        action_row = QHBoxLayout()
        restore = QPushButton("Restore example layout")
        restore.clicked.connect(self.restore_default)
        save = QPushButton("Save and apply")
        save.clicked.connect(self._commit)
        action_row.addWidget(restore)
        action_row.addWidget(save)
        layout.addLayout(action_row)
        note = QLabel(
            "Assign 'support' only if a genuine removable-support printhead "
            "and compatible material are eventually installed and verified. "
            "The existing Gongzheng channels are NOT presumed water-soluble.")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()
        self._rebuild()

    def _number(self, value, positive=False):
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(.1 if positive else -1000, 300 if positive else 1000)
        spin.setValue(float(value))
        spin.setSingleStep(.5)
        spin.setSuffix(" mm")
        spin.setKeyboardTracking(False)
        spin.valueChanged.connect(self._commit)
        return spin

    def _check(self, enabled):
        check = QCheckBox()
        check.setChecked(bool(enabled))
        check.toggled.connect(self._commit)
        return check

    def _name(self, value):
        item = QTableWidgetItem(str(value))
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        return item

    def _rebuild(self):
        self._loading = True
        try:
            self.head_table.setRowCount(0)
            for head in self.config.heads:
                row = self.head_table.rowCount()
                self.head_table.insertRow(row)
                self.head_table.setItem(row, 0, self._name(head.name))
                combo = QComboBox()
                for material in MATERIALS:
                    label = "Color (CMYK)" if material == "color" else material.title()
                    if material == "support":
                        label += " (experimental)"
                    combo.addItem(label, material)
                combo.setCurrentIndex(MATERIALS.index(head.material))
                combo.currentIndexChanged.connect(self._commit)
                self.head_table.setCellWidget(row, 1, combo)
                for col, value, positive in (
                    (2, head.offset_x_mm, False),
                    (3, head.offset_y_mm, False),
                    (4, head.width_mm, True),
                    (5, head.depth_mm, True),
                    (6, head.height_mm, True),
                    (7, head.swath_mm, True)):
                    self.head_table.setCellWidget(
                        row, col, self._number(value, positive))
                self.head_table.setCellWidget(row, 8, self._check(head.enabled))
            self.lamp_table.setRowCount(0)
            for lamp in self.config.lamps:
                row = self.lamp_table.rowCount()
                self.lamp_table.insertRow(row)
                self.lamp_table.setItem(row, 0, self._name(lamp.name))
                for col, value, positive in (
                    (1, lamp.offset_x_mm, False),
                    (2, lamp.offset_y_mm, False),
                    (3, lamp.width_mm, True),
                    (4, lamp.depth_mm, True),
                    (5, lamp.body_height_mm, True)):
                    self.lamp_table.setCellWidget(
                        row, col, self._number(value, positive))
                self.lamp_table.setCellWidget(row, 6, self._check(lamp.enabled))
            self.cure_passes.setValue(self.config.cure_passes)
        finally:
            self._loading = False
        self.status.setText(
            f"Layout stored at {gantry_file(self.root)}. "
            "Edit a value to update the virtual gantry and estimated time.")

    def _get_config(self):
        heads = []
        for row in range(self.head_table.rowCount()):
            heads.append(HeadConfig(
                name=self.head_table.item(row, 0).text(),
                material=self.head_table.cellWidget(row, 1).currentData(),
                offset_x_mm=self.head_table.cellWidget(row, 2).value(),
                offset_y_mm=self.head_table.cellWidget(row, 3).value(),
                width_mm=self.head_table.cellWidget(row, 4).value(),
                depth_mm=self.head_table.cellWidget(row, 5).value(),
                height_mm=self.head_table.cellWidget(row, 6).value(),
                swath_mm=self.head_table.cellWidget(row, 7).value(),
                enabled=self.head_table.cellWidget(row, 8).isChecked()))
        lamps = []
        for row in range(self.lamp_table.rowCount()):
            lamps.append(LampConfig(
                name=self.lamp_table.item(row, 0).text(),
                offset_x_mm=self.lamp_table.cellWidget(row, 1).value(),
                offset_y_mm=self.lamp_table.cellWidget(row, 2).value(),
                width_mm=self.lamp_table.cellWidget(row, 3).value(),
                depth_mm=self.lamp_table.cellWidget(row, 4).value(),
                body_height_mm=self.lamp_table.cellWidget(row, 5).value(),
                enabled=self.lamp_table.cellWidget(row, 6).isChecked()))
        result = GantryConfig(heads, lamps, self.cure_passes.value())
        result.validate()
        return result

    def _commit(self, *args):
        if self._loading:
            return
        try:
            config = self._get_config()
            save_gantry(self.root, config)
        except (OSError, ValueError, TypeError) as exc:
            self.status.setText(f"Gantry change not applied: {exc}")
            return
        self.config = config
        self.configurationChanged.emit(config)
        active_heads = sum(h.enabled for h in config.heads)
        active_lamps = sum(l.enabled for l in config.lamps)
        self.status.setText(
            f"Saved. Active: {active_heads} printheads, "
            f"{active_lamps} UV lamps. Virtual path and ETA updated.")

    def add_head(self):
        if self.head_table.rowCount() >= 12:
            self.status.setText("Maximum 12 printheads.")
            return
        self.config = self._get_config()
        num = len(self.config.heads) + 1
        offset = max((h.offset_x_mm for h in self.config.heads), default=-70)
        self.config.heads.append(
            HeadConfig(f"Head {num}", "support", offset + 70, 0))
        self._rebuild()
        self._commit()

    def remove_head(self):
        row = self.head_table.currentRow()
        if row < 0:
            return
        config = self._get_config()
        if sum(h.enabled for h in config.heads) <= 1 and config.heads[row].enabled:
            self.status.setText("Keep at least one enabled printhead.")
            return
        config.heads.pop(row)
        self.config = config
        self._rebuild()
        self._commit()

    def add_lamp(self):
        if self.lamp_table.rowCount() >= 12:
            self.status.setText("Maximum 12 lamps.")
            return
        self.config = self._get_config()
        num = len(self.config.lamps) + 1
        x = max((l.offset_x_mm for l in self.config.lamps), default=0)
        self.config.lamps.append(
            LampConfig(f"UV lamp {num}", x + 65, 0, 40, 45))
        self._rebuild()
        self._commit()

    def remove_lamp(self):
        row = self.lamp_table.currentRow()
        if row < 0:
            return
        config = self._get_config()
        config.lamps.pop(row)
        self.config = config
        self._rebuild()
        self._commit()

    def restore_default(self):
        self.config = default_gantry()
        self._rebuild()
        self._commit()
