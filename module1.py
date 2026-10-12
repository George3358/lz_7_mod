#!/usr/bin/env python3
"""
Road Traffic Editor
Оптимизированная версия с компактной кодовой базой и сохранением полного UI.
"""
from __future__ import annotations

import sys
import os
import json
import logging
import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

# Импорт Keras
KERAS_AVAILABLE = False
try:
    from keras.models import Sequential, load_model
    from keras.layers import Dense
    from keras.utils import to_categorical
    KERAS_AVAILABLE = True
except Exception:
    try:
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.utils import to_categorical
        KERAS_AVAILABLE = True
    except Exception as err:
        logging.error(f"Keras import error: {err}")

BASE_PATH = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsScene, QGraphicsView,
    QGraphicsItem, QGraphicsPixmapItem, QGraphicsTextItem,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,
    QFormLayout, QDockWidget, QLabel, QFileDialog, QGroupBox,
    QStatusBar, QAction, QMessageBox, QListWidget, QListWidgetItem,
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QPen, QColor, QPixmap, QTransform, QIcon, QKeyEvent, QPainter, QCursor
from PyQt5.QtCore import Qt, QPoint, QPointF, pyqtSignal, QTimer

DEFAULT_GRID_SIZE = 35
SUPPORTED_IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.bmp', '.svg', '.ico']
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger('RoadEditor')


# ──────────────────────────────────────────────────────────────
# База данных
# ──────────────────────────────────────────────────────────────
class DatabaseManager:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or (BASE_PATH / 'traffic_events.db'))
        self.init_db()

    def _execute(self, query: str, params: tuple = (), fetch: bool = False):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            res = cur.fetchall() if fetch else None
            conn.commit()
            return res

    def init_db(self):
        self._execute('''CREATE TABLE IF NOT EXISTS system_events 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, event_type TEXT, event_description TEXT, status TEXT)''')
        self._execute('''CREATE TABLE IF NOT EXISTS training_process 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, epoch INTEGER, loss REAL, accuracy REAL, timestamp TEXT)''')
        self._execute('''CREATE TABLE IF NOT EXISTS training_events 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, start_time TEXT, epoch INTEGER, model_saved TEXT, status TEXT)''')

    def log_event(self, ev_type: str, desc: str, status: str = "Успех"):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._execute("INSERT INTO system_events (timestamp, event_type, event_description, status) VALUES (?, ?, ?, ?)",
                      (t, ev_type, desc, status))

    def log_training_step(self, epoch: int, loss: float, acc: float, start_time: str, model_file: str):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._execute("INSERT INTO training_process (epoch, loss, accuracy, timestamp) VALUES (?, ?, ?, ?)",
                      (epoch, float(loss), float(acc), t))
        self._execute("INSERT INTO training_events (start_time, epoch, model_saved, status) VALUES (?, ?, ?, ?)",
                      (start_time, epoch, model_file, "Успех"))

    def get_training_process(self) -> List[tuple]:
        return self._execute("SELECT epoch, loss, accuracy, timestamp FROM training_process ORDER BY id ASC", fetch=True)

    def get_training_events(self) -> List[tuple]:
        return self._execute("SELECT start_time, epoch, model_saved, status FROM training_events ORDER BY id DESC", fetch=True)

    def get_last_epoch(self) -> int:
        r = self._execute("SELECT MAX(epoch) FROM training_process", fetch=True)
        return r[0][0] if r and r[0][0] is not None else 0

    def get_all_events(self) -> List[tuple]:
        return self._execute("SELECT * FROM system_events ORDER BY timestamp DESC", fetch=True)

    def clear_events(self):
        self._execute("DELETE FROM system_events")


# ──────────────────────────────────────────────────────────────
# Модель нейросети
# ──────────────────────────────────────────────────────────────
class TrafficModelManager:
    def __init__(self, data_db: str = "data.db", model_path: str = "traffic_model.keras"):
        self.data_db = BASE_PATH / data_db
        self.model_path = BASE_PATH / model_path
        self.model = None

    def ensure_labeled_data(self):
        with sqlite3.connect(str(self.data_db)) as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS traffic 
                (id INTEGER PRIMARY KEY AUTOINCREMENT, temperature INTEGER, cars INTEGER, light INTEGER, traffic_level INTEGER)""")
            try:
                cur.execute("ALTER TABLE traffic ADD COLUMN traffic_level INTEGER")
            except sqlite3.OperationalError:
                pass

            cur.execute("SELECT COUNT(*) FROM traffic")
            if cur.fetchone()[0] == 0:
                import random
                for _ in range(1000):
                    t, c, l = random.randint(-30, 40), random.randint(0, 80), random.randint(0, 1)
                    wf = 1.5 if (t < -10 or t > 35) else 1.0
                    lf = 1.2 if l == 1 else 0.7
                    lvl = 0 if (c * wf * lf) < 20 else (1 if (c * wf * lf) < 50 else 2)
                    cur.execute("INSERT INTO traffic (temperature, cars, light, traffic_level) VALUES (?, ?, ?, ?)", (t, c, l, lvl))
            else:
                cur.execute("SELECT id, temperature, cars, light FROM traffic WHERE traffic_level IS NULL")
                for rid, t, c, l in cur.fetchall():
                    wf = 1.5 if (t < -10 or t > 35) else 1.0
                    lf = 1.2 if l == 1 else 0.7
                    lvl = 0 if (c * wf * lf) < 20 else (1 if (c * wf * lf) < 50 else 2)
                    cur.execute("UPDATE traffic SET traffic_level = ? WHERE id = ?", (lvl, rid))
            conn.commit()

    def load_data(self):
        self.ensure_labeled_data()
        with sqlite3.connect(str(self.data_db)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT temperature, cars, light, traffic_level FROM traffic")
            rows = cur.fetchall()

        X = np.array([[r[0], r[1], r[2]] for r in rows], dtype=float)
        Y = np.array([r[3] for r in rows], dtype=int)
        X[:, 0] = (X[:, 0] + 30) / 70.0
        X[:, 1] = X[:, 1] / 80.0
        return X, to_categorical(Y, num_classes=3)

    def get_or_create_model(self):
        if self.model:
            return self.model
        if self.model_path.exists():
            try:
                self.model = load_model(str(self.model_path))
                return self.model
            except Exception:
                pass
        self.model = Sequential([
            Dense(16, activation='relu', input_shape=(3,)),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.model

    def train_one_epoch(self) -> Tuple[float, float, str]:
        if not KERAS_AVAILABLE:
            raise RuntimeError("Keras недоступен.")
        X, Y = self.load_data()
        m = self.get_or_create_model()
        h = m.fit(X, Y, epochs=1, batch_size=32, verbose=0)
        m.save(str(self.model_path))
        return h.history['loss'][0], h.history['accuracy'][0], self.model_path.name

    def predict_current_traffic(self, t: int, cars: int, light: int):
        if not KERAS_AVAILABLE:
            return None, "Keras недоступен", 0.0
        try:
            m = self.get_or_create_model()
            inp = np.array([[(t + 30) / 70.0, cars / 80.0, light]], dtype=float)
            probs = m.predict(inp, verbose=0)[0]
            idx = int(np.argmax(probs))
            lbls = {0: "0 (свободно)", 1: "1 (средняя загруженность)", 2: "2 (затор / пробка)"}
            return idx, lbls.get(idx, "Не определено"), float(probs[idx])
        except Exception as e:
            return None, f"Ошибка: {e}", 0.0


# ──────────────────────────────────────────────────────────────
# Универсальное окно таблицы со сводкой
# ──────────────────────────────────────────────────────────────
class TrainingTableDialog(QDialog):
    def __init__(self, title: str, headers: List[str], data: List[tuple], summary_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(650, 480)
        layout = QVBoxLayout(self)

        grp = QGroupBox("Текущая загруженность перекрестка")
        grp_l = QVBoxLayout(grp)
        lbl = QLabel(summary_text)
        lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #1565c0;")
        grp_l.addWidget(lbl)
        layout.addWidget(grp)

        self.table = QTableWidget(len(data), len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for r_idx, row in enumerate(data):
            for c_idx, val in enumerate(row):
                v_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                self.table.setItem(r_idx, c_idx, QTableWidgetItem(v_str))
        layout.addWidget(self.table)

        btn = QPushButton("Закрыть")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


# ──────────────────────────────────────────────────────────────
# Контроллер светофора
# ──────────────────────────────────────────────────────────────
class TrafficLightController:
    def __init__(self, item, db_manager: DatabaseManager, num: int = 1):
        self.item = item
        self.db = db_manager
        self.light_number = num
        self.mode = "manual"
        self.current_state = "red"
        self.trap_active = False

        self.auto_timer = QTimer()
        self.auto_timer.timeout.connect(self._auto_cycle)
        self.trap_timer = QTimer()
        self.trap_timer.setSingleShot(True)
        self.trap_timer.timeout.connect(self.recovery)

    def set_manual_mode(self):
        self.mode = "manual"
        self.auto_timer.stop()
        self.db.log_event("Режим", f"Светофор #{self.light_number} -> ручной")

    def set_auto_mode(self):
        self.mode = "auto"
        self.auto_timer.start(10000)
        self.db.log_event("Режим", f"Светофор #{self.light_number} -> авто")

    def set_state(self, state: str):
        if self.trap_active:
            return False
        self.current_state = state
        self.item.props["State"] = state
        self.item.update_visuals()
        self.db.log_event("Ручное управление", f"Светофор #{self.light_number}: {state}")
        return True

    def _auto_cycle(self):
        if self.trap_active:
            return
        nxt = {"red": ("yellow", 5000), "yellow": ("green", 10000), "green": ("red", 10000)}
        state, interval = nxt.get(self.current_state, ("red", 10000))
        self.set_state(state)
        self.auto_timer.setInterval(interval)

    def activate_trap(self):
        self.trap_active = True
        self.auto_timer.stop()
        self.set_state("red")
        self.db.log_event("Ошибка", f"Светофор #{self.light_number}: Сбой", "Ошибка")

    def recovery(self):
        if self.trap_active:
            self.trap_active = False
            self.set_state("red")
            self.db.log_event("Восстановление", f"Светофор #{self.light_number} восстановлен")
        if self.mode == "auto":
            self.auto_timer.start(10000)


# ──────────────────────────────────────────────────────────────
# Ресурсы сцены и GridView
# ──────────────────────────────────────────────────────────────
class Config:
    def __init__(self, media_dir: Optional[Path] = None):
        self.media_dir = media_dir or (BASE_PATH / 'media')
        self.grid_size = DEFAULT_GRID_SIZE
        self.scene_width, self.scene_height = 1000, 800
        self.media_dir.mkdir(parents=True, exist_ok=True)

    def find_image(self, base_name: str) -> Optional[Path]:
        p = self.media_dir / base_name
        if p.is_file():
            return p
        for ext in SUPPORTED_IMAGE_EXTS:
            cp = self.media_dir / f"{base_name}{ext}"
            if cp.is_file():
                return cp
        return None

    def find_car_image(self, color: str, direction: str) -> Optional[Path]:
        dir_map = {"vertical": "Vertical", "horizontal": "Horizontal", "left": "Left", "right": "Right", "bottom": "Bottom"}
        sfx = dir_map.get(direction, "Vertical")
        pref = ["BC", "bc", "BlueCar", "B"] if color == "blue" else ["C", "c", "RedCar", "Car"]
        for p in pref:
            res = self.find_image(f"{p}{sfx}")
            if res:
                return res
        return None


class GridView(QGraphicsView):
    coords_updated = pyqtSignal(int, int)

    def __init__(self, scene, editor, config):
        super().__init__(scene)
        self.editor = editor
        self.config = config
        self.setMouseTracking(True)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setCursor(Qt.CrossCursor)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self._coord_label = None

    def _ensure_coord_label(self):
        """Гарантирует существование объекта метки на сцене без вылетов C++"""
        if self._coord_label is not None:
            try:
                if self._coord_label.scene() == self.scene():
                    return self._coord_label
            except RuntimeError:
                pass
        self._coord_label = QGraphicsTextItem()
        self._coord_label.setZValue(1000)
        self._coord_label.setDefaultTextColor(Qt.red)
        self._coord_label.hide()
        self.scene().addItem(self._coord_label)
        return self._coord_label

    def _snap(self, pos: QPointF) -> QPoint:
        gs = self.config.grid_size
        return QPoint(int(pos.x() // gs) * gs, int(pos.y() // gs) * gs)

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        painter.setPen(QPen(QColor(220, 220, 220), 1, Qt.DotLine))
        gs = self.config.grid_size
        l = int(rect.left()) - (int(rect.left()) % gs)
        r = int(rect.right()) + gs
        t = int(rect.top()) - (int(rect.top()) % gs)
        b = int(rect.bottom()) + gs
        for x in range(l, r, gs):
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
        for y in range(t, b, gs):
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)

    def mouseMoveEvent(self, event):
        sp = self._snap(self.mapToScene(event.pos()))
        try:
            lbl = self._ensure_coord_label()
            lbl.setPlainText(f"({sp.x()}, {sp.y()})")
            lbl.setPos(sp.x() + 12, sp.y() - 25)
            lbl.show()
        except RuntimeError:
            pass
        self.coords_updated.emit(sp.x(), sp.y())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        try:
            if self._coord_label is not None:
                self._coord_label.hide()
        except RuntimeError:
            self._coord_label = None
        self.coords_updated.emit(-1, -1)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.editor.selected_obj_type:
            sp = self._snap(self.mapToScene(event.pos()))
            self.editor.add_object(sp.x(), sp.y(), self.editor.selected_obj_type)
            self.editor.selected_obj_type = None
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        for item in self.scene().selectedItems():
            if isinstance(item, SceneObject) and item.flags() & QGraphicsItem.ItemIsMovable:
                item.setPos(self._snap(item.pos()))
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            f = 1.15 if event.angleDelta().y() > 0 else 0.85
            self.scale(f, f)
            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        mod, k = event.modifiers(), event.key()
        if mod & Qt.ControlModifier and k == Qt.Key_C:
            self.editor.copy_objects()
            return
        if mod & Qt.ControlModifier and k == Qt.Key_V:
            self.editor.paste_objects()
            return
        if k == Qt.Key_R:
            self.editor.rotate_selected()
            return
        if k in (Qt.Key_Delete, Qt.Key_Backspace):
            for i in [x for x in self.scene().selectedItems() if isinstance(x, SceneObject)]:
                self.editor.delete_obj(i)
            return
        d = {Qt.Key_Up: (0, -35), Qt.Key_Down: (0, 35), Qt.Key_Left: (-35, 0), Qt.Key_Right: (35, 0)}.get(k)
        if d and not mod:
            for item in [x for x in self.scene().selectedItems() if isinstance(x, SceneObject)]:
                item.setPos(item.pos().x() + d[0], item.pos().y() + d[1])
            return
        super().keyPressEvent(event)


# ──────────────────────────────────────────────────────────────
# Графический объект сцены
# ──────────────────────────────────────────────────────────────
class SceneObject(QGraphicsPixmapItem):
    def __init__(self, x, y, obj_type, props=None, config=None):
        super().__init__()
        self.obj_type, self.props = obj_type, props or {}
        self.config = config or Config()
        self.setPos(x, y)
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
        self.coord_label = QGraphicsTextItem(self) if self.obj_type.startswith("R") else None
        if self.coord_label:
            self.coord_label.setDefaultTextColor(QColor(255, 200, 0))
            self.coord_label.setPos(0, -22)
            self.coord_label.hide()
        self.update_visuals()

    def update_visuals(self):
        pix = QPixmap()
        cfg = self.config
        path = None
        if self.obj_type == "Rvertical":
            path = cfg.find_image("Rvertical")
            if path and pix.load(str(path)):
                r = int(self.props.get("Rotation", 0))
                if r: pix = pix.transformed(QTransform().rotate(r))
        elif self.obj_type == "Rcrossroads":
            path = cfg.find_image("Rcrossroads")
            if path: pix.load(str(path))
        elif self.obj_type == "светофор":
            path = cfg.find_image(f"TL{self.props.get('State', 'red')}")
            if path: pix.load(str(path))
        elif self.obj_type == "пешеходный переход":
            path = cfg.find_image(f"Z{self.props.get('Direction', 'vertical')}")
            if path: pix.load(str(path))
        elif self.obj_type == "пешеход":
            path = cfg.find_image("Pedestrian") or cfg.find_image("Pedestrain")
            if path: pix.load(str(path))
        elif self.obj_type == "движение запрещено":
            path = cfg.find_image(self.props.get("Type", "Stop"))
            if path: pix.load(str(path))
        elif self.obj_type.startswith("car_"):
            p = self.obj_type.split("_")
            c, d = (p[1], p[2]) if len(p) > 2 else ("red", "vertical")
            path = cfg.find_car_image(c, d)
            if path and pix.load(str(path)):
                rot = {"vertical": 0, "right": 90, "horizontal": 90, "bottom": 180, "left": 270}.get(d, 0)
                if rot and d not in path.name.lower():
                    pix = pix.transformed(QTransform().rotate(rot))

        if pix.isNull():
            pix = QPixmap(35, 35)
            pix.fill(QColor(120, 120, 120, 180))
        self.setPixmap(pix)
        self.setZValue(10 if self.obj_type.startswith("car_") else -18)
        if self.coord_label:
            self.coord_label.setPlainText(f"({int(self.x())}, {int(self.y())})")

    def hoverEnterEvent(self, e):
        if self.coord_label: self.coord_label.show()
        super().hoverEnterEvent(e)

    def hoverLeaveEvent(self, e):
        if self.coord_label: self.coord_label.hide()
        super().hoverLeaveEvent(e)

    def to_dict(self):
        return {"x": self.x(), "y": self.y(), "obj_type": self.obj_type, "props": self.props}


# ──────────────────────────────────────────────────────────────
# Главное окно RoadEditor
# ──────────────────────────────────────────────────────────────
class RoadEditor(QMainWindow):
    def __init__(self, config, initial_scene=None):
        super().__init__()
        self.config = config
        self.setWindowTitle("Road Traffic Editor")
        self.setGeometry(100, 100, 1400, 900)

        self.db_manager = DatabaseManager()
        self.model_manager = TrafficModelManager()
        self.traffic_light_controllers = {}
        self.traffic_light_counter = 0
        self.clipboard_data = []
        self.last_copy_pos = QPoint(0, 0)
        self.selected_obj_type = None

        self.scene = QGraphicsScene(0, 0, config.scene_width, config.scene_height)
        self.view = GridView(self.scene, self, config)
        self.view.coords_updated.connect(lambda x, y: self.coords_label.setText(f"Курсор: ({x}, {y})" if x >= 0 else ""))

        self.setCentralWidget(self.view)
        self._init_menu_and_status()
        self._create_toolbar()
        self._create_properties_panel()
        self._create_control_panel()

        if initial_scene and initial_scene.exists():
            self.load_scene_from_file(initial_scene)

    def _init_menu_and_status(self):
        mb = self.menuBar()
        fm = mb.addMenu("Файл")
        fm.addAction("Новая сцена", self._clear_editor, "Ctrl+N")
        fm.addAction("Открыть...", self.load_scene, "Ctrl+O")
        fm.addAction("Сохранить", self.save_scene, "Ctrl+S")
        fm.addAction("Очистить редактор", self._clear_editor, "Ctrl+Shift+C")
        fm.addSeparator()
        fm.addAction("Выход", self.close, "Ctrl+Q")

        dm = mb.addMenu("База данных")
        dm.addAction("Просмотр событий", self._view_database)
        dm.addAction("Очистить базу", self.db_manager.clear_events)

        mb.addMenu("Справка").addAction("О программе", lambda: QMessageBox.about(self, "О программе", "Road Traffic Editor"))

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.coords_label = QLabel("")
        self.statusBar.addWidget(QLabel("Готов"))
        self.statusBar.addPermanentWidget(self.coords_label)

    def _create_toolbar(self):
        tb = QDockWidget("Инструменты", self)
        self.addDockWidget(Qt.LeftDockWidgetArea, tb)
        w = QWidget()
        lay = QVBoxLayout(w)

        # Блок нейросети
        nn_grp = QGroupBox("Нейросеть")
        nn_l = QVBoxLayout(nn_grp)
        b_train = QPushButton("Начать обучение")
        b_train.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        b_train.clicked.connect(self._on_train_single_epoch)
        b_proc = QPushButton("Процесс обучения")
        b_proc.clicked.connect(self._on_show_training_process)
        b_ev = QPushButton("Журнал событий")
        b_ev.clicked.connect(self._on_show_training_events)
        for b in (b_train, b_proc, b_ev): nn_l.addWidget(b)
        lay.addWidget(nn_grp)

        # Блок объектов
        obj_grp = QGroupBox("Добавить объекты")
        obj_l = QVBoxLayout(obj_grp)
        objs = [
            ("Светофор", "светофор"), ("Пешеходный переход", "пешеходный переход"),
            ("Пешеход", "пешеход"), ("Знак", "движение запрещено"),
            ("Дорога (верт.)", "Rvertical"), ("Перекресток", "Rcrossroads"),
            ("Машина (Красная)", "car_red_vertical"), ("Машина (Синяя)", "car_blue_vertical")
        ]
        for name, t in objs:
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, typ=t: setattr(self, 'selected_obj_type', typ))
            obj_l.addWidget(btn)
        lay.addWidget(obj_grp)

        f_grp = QGroupBox("Файл")
        fl = QVBoxLayout(f_grp)
        for name, fn in [("Сохранить", self.save_scene), ("Загрузить", self.load_scene), ("Очистить", self._clear_editor)]:
            b = QPushButton(name); b.clicked.connect(fn); fl.addWidget(b)
        lay.addWidget(f_grp)

        lay.addStretch()
        tb.setWidget(w)

    def _create_properties_panel(self):
        self.prop_dock = QDockWidget("Свойства объекта", self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.prop_dock)
        self.prop_container = QWidget()
        self.prop_layout = QFormLayout(self.prop_container)
        self.prop_dock.setWidget(self.prop_container)
        self.scene.selectionChanged.connect(self._update_properties_panel)
        self._update_properties_panel()

    def _create_control_panel(self):
        dock = QDockWidget("Панель управления", self)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget()
        lay = QVBoxLayout(w)

        self.tl_list = QListWidget()
        self.tl_list.itemClicked.connect(self._on_tl_selected)
        lay.addWidget(QLabel("Список светофоров:"))
        lay.addWidget(self.tl_list)

        del_btn = QPushButton("Удалить выбранный светофор")
        del_btn.setStyleSheet("background-color: #ff4444; color: white;")
        del_btn.clicked.connect(self._delete_selected_tl)
        lay.addWidget(del_btn)

        self.ctrl_grp = QGroupBox("Управление выбранным светофором")
        self.ctrl_grp.setEnabled(False)
        cl = QVBoxLayout(self.ctrl_grp)
        self.tl_info_lbl = QLabel("Не выбран")
        cl.addWidget(self.tl_info_lbl)

        # Ручное
        m_grp = QGroupBox("Ручное управление")
        ml = QHBoxLayout(m_grp)
        for s, col in [("Красный", "red"), ("Желтый", "yellow"), ("Зеленый", "green")]:
            b = QPushButton(s); b.clicked.connect(lambda _, st=col: self._set_tl_state(st)); ml.addWidget(b)
        cl.addWidget(m_grp)

        # Режим
        rm_grp = QGroupBox("Режим работы")
        rml = QHBoxLayout(rm_grp)
        self.b_man = QPushButton("Ручной режим"); self.b_man.clicked.connect(lambda: self._set_tl_mode("manual"))
        self.b_aut = QPushButton("Автоматический"); self.b_aut.clicked.connect(lambda: self._set_tl_mode("auto"))
        rml.addWidget(self.b_man); rml.addWidget(self.b_aut)
        cl.addWidget(rm_grp)

        # Авария
        em_grp = QGroupBox("Аварийное управление")
        eml = QHBoxLayout(em_grp)
        b_trap = QPushButton("Воспроизведение ошибки"); b_trap.setStyleSheet("background-color: #ff4444; color: white;")
        b_trap.clicked.connect(self._activate_trap)
        b_rec = QPushButton("Сброс аварии"); b_rec.setStyleSheet("background-color: #44ff44;")
        b_rec.clicked.connect(self._recovery_trap)
        eml.addWidget(b_trap); eml.addWidget(b_rec)
        cl.addWidget(em_grp)

        lay.addWidget(self.ctrl_grp)
        lay.addStretch()
        dock.setWidget(w)
        self.scene.sceneRectChanged.connect(self._update_traffic_light_list)

    # ── Логика таблиц и обучения ──
    def _get_traffic_summary(self) -> str:
        cars = sum(1 for i in self.scene.items() if isinstance(i, SceneObject) and i.obj_type.startswith("car_"))
        _, pred_t, conf = self.model_manager.predict_current_traffic(20, cars, 1)
        return f"Машин на схеме: {cars} | Температура: 20°C | Освещенность: день\nУровень нагрузки: {pred_t} (уверенность: {conf:.1%})"

    def _on_train_single_epoch(self):
        if not KERAS_AVAILABLE:
            QMessageBox.critical(self, "Ошибка", "Keras недоступен.")
            return
        t_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ep = self.db_manager.get_last_epoch() + 1
        try:
            self.statusBar.showMessage(f"Обучение эпохи #{ep}...", 1500)
            QApplication.processEvents()
            loss, acc, m_name = self.model_manager.train_one_epoch()
            self.db_manager.log_training_step(ep, loss, acc, t_start, m_name)
            QMessageBox.information(self, "Обучение", f"Эпоха #{ep} завершена!\nLoss: {loss:.4f} | Accuracy: {acc:.4f}\nСохранено в {m_name}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def _on_show_training_process(self):
        TrainingTableDialog("Процесс обучения", ["Эпоха", "Loss", "Accuracy", "Время"],
                            self.db_manager.get_training_process(), self._get_traffic_summary(), self).exec_()

    def _on_show_training_events(self):
        TrainingTableDialog("Журнал событий", ["Время старта", "Эпоха", "Файл модели", "Статус"],
                            self.db_manager.get_training_events(), self._get_traffic_summary(), self).exec_()

    # ── Управление объектами ──
    def add_object(self, x, y, obj_type):
        defs = {"Rvertical": {"Rotation": 0}, "светофор": {"State": "red"}, "пешеходный переход": {"Direction": "vertical"},
                "движение запрещено": {"Type": "Stop"}, "car_red_vertical": {"Direction": "vertical"}, "car_blue_vertical": {"Direction": "vertical"}}
        item = SceneObject(x, y, obj_type, defs.get(obj_type, {}).copy(), self.config)
        self.scene.addItem(item)
        if obj_type == "светофор":
            self.traffic_light_counter += 1
            self.traffic_light_controllers[id(item)] = TrafficLightController(item, self.db_manager, self.traffic_light_counter)
            self._update_traffic_light_list()
        self.statusBar.showMessage(f"Добавлен: {obj_type}", 1500)

    def delete_obj(self, item):
        if item.obj_type == "светофор":
            ctrl = self.traffic_light_controllers.pop(id(item), None)
            if ctrl: ctrl.auto_timer.stop()
            self._update_traffic_light_list()
        self.scene.removeItem(item)

    def _update_properties_panel(self):
        while self.prop_layout.count():
            w = self.prop_layout.takeAt(0).widget()
            if w: w.deleteLater()
        items = [i for i in self.scene.selectedItems() if isinstance(i, SceneObject)]
        if not items:
            self.prop_layout.addRow(QLabel("<i>Объект не выбран</i>"))
            return
        it = items[0]
        self.prop_layout.addRow("Координаты:", QLabel(f"({int(it.x())}, {int(it.y())})"))
        if it.obj_type == "светофор":
            c = QComboBox(); c.addItems(["red", "yellow", "green"]); c.setCurrentText(it.props.get("State", "red"))
            c.currentTextChanged.connect(lambda v: (it.props.update({"State": v}), it.update_visuals()))
            self.prop_layout.addRow("Состояние:", c)
        elif it.obj_type == "Rvertical":
            c = QComboBox(); c.addItems(["0", "90", "180", "270"]); c.setCurrentText(str(it.props.get("Rotation", 0)))
            c.currentTextChanged.connect(lambda v: (it.props.update({"Rotation": int(v)}), it.update_visuals()))
            self.prop_layout.addRow("Поворот:", c)
        elif it.obj_type.startswith("car_"):
            c = QComboBox(); c.addItems(["vertical", "right", "bottom", "left"])
            p = it.obj_type.split("_")
            c.setCurrentText(p[2] if len(p) > 2 else "vertical")
            c.currentTextChanged.connect(lambda v: setattr(it, 'obj_type', f"car_{p[1]}_{v}") or it.update_visuals())
            self.prop_layout.addRow("Направление:", c)
        btn = QPushButton("Удалить объект"); btn.setStyleSheet("color: red;")
        btn.clicked.connect(lambda: self.delete_obj(it))
        self.prop_layout.addRow(btn)

    def _update_traffic_light_list(self):
        self.tl_list.clear()
        self.traffic_light_controllers.clear()
        self.traffic_light_counter = 0
        lights = sorted([i for i in self.scene.items() if isinstance(i, SceneObject) and i.obj_type == "светофор"],
                        key=lambda x: (x.y(), x.x()))
        for item in lights:
            self.traffic_light_counter += 1
            self.traffic_light_controllers[id(item)] = TrafficLightController(item, self.db_manager, self.traffic_light_counter)
            li = QListWidgetItem(f"Светофор #{self.traffic_light_counter} ({int(item.x())}, {int(item.y())})")
            li.setData(Qt.UserRole, id(item))
            self.tl_list.addItem(li)

    def _on_tl_selected(self, item):
        ctrl = self.traffic_light_controllers.get(item.data(Qt.UserRole))
        if ctrl:
            self.scene.clearSelection()
            ctrl.item.setSelected(True)
            self.tl_info_lbl.setText(f"#{ctrl.light_number} ({int(ctrl.item.x())}, {int(ctrl.item.y())}) | {ctrl.mode} | {ctrl.current_state}")
            self.ctrl_grp.setEnabled(True)
            self.b_man.setStyleSheet("background-color: #44ff44;" if ctrl.mode == "manual" else "")
            self.b_aut.setStyleSheet("background-color: #44ff44;" if ctrl.mode == "auto" else "")

    def _set_tl_state(self, st):
        its = self.tl_list.selectedItems()
        if its:
            c = self.traffic_light_controllers.get(its[0].data(Qt.UserRole))
            if c and c.set_state(st): self._on_tl_selected(its[0])

    def _set_tl_mode(self, m):
        its = self.tl_list.selectedItems()
        if its:
            c = self.traffic_light_controllers.get(its[0].data(Qt.UserRole))
            if c:
                c.set_manual_mode() if m == "manual" else c.set_auto_mode()
                self._on_tl_selected(its[0])

    def _activate_trap(self):
        its = self.tl_list.selectedItems()
        if its and QMessageBox.question(self, "Авария", "Активировать сбой?") == QMessageBox.Yes:
            c = self.traffic_light_controllers.get(its[0].data(Qt.UserRole))
            if c: c.activate_trap(); self._on_tl_selected(its[0])

    def _recovery_trap(self):
        its = self.tl_list.selectedItems()
        if its:
            c = self.traffic_light_controllers.get(its[0].data(Qt.UserRole))
            if c: c.recovery(); self._on_tl_selected(its[0])

    def _delete_selected_tl(self):
        its = self.tl_list.selectedItems()
        if its:
            c = self.traffic_light_controllers.get(its[0].data(Qt.UserRole))
            if c: self.delete_obj(c.item)

    def copy_objects(self):
        sel = [i for i in self.scene.selectedItems() if isinstance(i, SceneObject)]
        if sel:
            self.clipboard_data = [i.to_dict() for i in sel]
            self.last_copy_pos = QPoint(int(sum(i.x() for i in sel) / len(sel)), int(sum(i.y() for i in sel) / len(sel)))

    def paste_objects(self):
        if not self.clipboard_data: return
        cur = self.view.mapToScene(self.view.mapFromGlobal(QCursor.pos()))
        bx, by = round(cur.x() / 35) * 35, round(cur.y() / 35) * 35
        self.scene.clearSelection()
        for d in self.clipboard_data:
            it = SceneObject(bx + (d["x"] - self.last_copy_pos.x()), by + (d["y"] - self.last_copy_pos.y()),
                             d["obj_type"], d.get("props", {}), self.config)
            self.scene.addItem(it)
            it.setSelected(True)
        self._update_traffic_light_list()

    def rotate_selected(self):
        for i in [x for x in self.scene.selectedItems() if isinstance(x, SceneObject) and x.obj_type == "Rvertical"]:
            i.props["Rotation"] = (i.props.get("Rotation", 0) + 90) % 360
            i.update_visuals()

    def save_scene(self):
        p, _ = QFileDialog.getSaveFileName(self, "Сохранить", "", "JSON (*.json)")
        if p:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump([i.to_dict() for i in self.scene.items() if isinstance(i, SceneObject)], f, indent=2)

    def load_scene(self):
        p, _ = QFileDialog.getOpenFileName(self, "Открыть", "", "JSON (*.json)")
        if p: self.load_scene_from_file(Path(p))

    def load_scene_from_file(self, path: Path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._clear_editor()
            for entry in data:
                it = SceneObject(entry["x"], entry["y"], entry["obj_type"], entry.get("props", {}), self.config)
                self.scene.addItem(it)
            self._update_traffic_light_list()
        except Exception as e:
            self.statusBar.showMessage(f"Ошибка загрузки: {e}", 3000)

    def _clear_editor(self):
        self.scene.clear()
        for c in self.traffic_light_controllers.values(): c.auto_timer.stop()
        self.traffic_light_controllers.clear()
        self.tl_list.clear()

    def _view_database(self):
        evs = self.db_manager.get_all_events()
        msg = "\n".join([f"{e[1]} | {e[2]} | {e[4]}\n{e[3]}" for e in evs[:15]]) if evs else "События отсутствуют"
        QMessageBox.information(self, "События", msg)


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--media', type=Path)
    args.add_argument('--scene', type=Path)
    p = args.parse_args()

    app = QApplication(sys.argv)
    w = RoadEditor(Config(media_dir=p.media), p.scene)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()