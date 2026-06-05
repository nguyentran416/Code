import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
keras = tf.keras
import os
import shutil
import logging
import base64
import json
import unicodedata
from datetime import datetime, timedelta
from collections import Counter
import re
from sqlalchemy import func, inspect, text
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
from flask import Flask, request, render_template, url_for, redirect, session, flash, abort
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

import tensorflow as tf
import tensorflow_hub as hub
import h5py
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "your_secret_key")

UPLOAD_MAX_MB = int(os.environ.get('UPLOAD_MAX_MB', '50'))
app.config['MAX_CONTENT_LENGTH'] = UPLOAD_MAX_MB * 1024 * 1024

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
STATIC_UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
NEWS_UPLOAD_FOLDER = os.path.join(STATIC_UPLOAD_FOLDER, 'news')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(STATIC_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(NEWS_UPLOAD_FOLDER, exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'waste_classification.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE_DIR, 'model_best.keras'))
app.config['MODEL_PATH'] = MODEL_PATH

# ============================================================
# DETECTION THRESHOLDS
# ============================================================
CLASSIFICATION_USE_TTA = os.environ.get('CLASSIFICATION_USE_TTA', 'true').lower() in ('1', 'true', 'yes')


# SSD object-ness score — tăng ngưỡng để giảm box nhiễu

# Loosened thresholds for better recall

# ==== Tối ưu nhận diện đa vật thể ====
DETECT_MIN_SCORE          = float(os.environ.get('DETECT_MIN_SCORE',          '0.38'))  # Giảm để bắt vật nhỏ
DETECT_MIN_LABEL_SCORE    = float(os.environ.get('DETECT_MIN_LABEL_SCORE',    '0.60'))  # Tăng confidence
DETECT_GRID_MIN_SCORE     = float(os.environ.get('DETECT_GRID_MIN_SCORE',     '0.55'))
DETECT_MIN_COMBINED_SCORE = float(os.environ.get('DETECT_MIN_COMBINED_SCORE', '0.25'))

# Diện tích box hợp lệ — giảm min, tăng max
DETECT_MIN_BOX_AREA = float(os.environ.get('DETECT_MIN_BOX_AREA', '0.04'))   # Tăng lên để loại box nhỏ, giảm tách nhỏ
DETECT_MAX_BOX_AREA = float(os.environ.get('DETECT_MAX_BOX_AREA', '0.92'))    # Cho phép vật lớn hơn

GRID_FALLBACK_ENABLED    = os.environ.get('GRID_FALLBACK_ENABLED', 'true').lower() in ('1', 'true', 'yes')
USE_SSD_DETECTION        = os.environ.get('USE_SSD_DETECTION', 'true').lower() in ('1', 'true', 'yes')
USE_SALIENCY_DETECTION   = os.environ.get('USE_SALIENCY_DETECTION', 'true').lower() in ('1', 'true', 'yes')

# Saliency — bổ sung khi SSD bỏ sót (ngưỡng thấp hơn)
SALIENCY_MIN_AREA_RATIO  = float(os.environ.get('SALIENCY_MIN_AREA_RATIO',  '0.010'))  # 1.0%
SALIENCY_MAX_AREA_RATIO  = float(os.environ.get('SALIENCY_MAX_AREA_RATIO',  '0.95'))   # tới 95%
SALIENCY_MIN_LABEL_SCORE = float(os.environ.get('SALIENCY_MIN_LABEL_SCORE', '0.45'))   # Thấp hơn

MAX_MULTI_DETECTIONS = int(os.environ.get('MAX_MULTI_DETECTIONS', '12'))  # Tăng lên 12
IMG_CLASSIFY_SIZE    = 224
TRIM_BOTTOM_RATIO    = float(os.environ.get('TRIM_BOTTOM_RATIO', '0.08'))

# NMS IoU
NMS_IOU_THRESHOLD = float(os.environ.get('NMS_IOU_THRESHOLD', '0.22'))  # Tăng NMS, giảm box chồng lấn

# Fallback: chỉ dùng full-image khi score detect thực sự rất thấp
SINGLE_DET_FALLBACK_SCORE = float(os.environ.get('SINGLE_DET_FALLBACK_SCORE', '0.40'))  # hạ từ 0.60

# Confidence tối thiểu để chấp nhận kết quả single
SINGLE_MIN_CONFIDENCE = float(os.environ.get('SINGLE_MIN_CONFIDENCE', '0.25'))  # hạ từ 0.30

# Aspect ratio — nới rộng để bắt vật dẹt/mỏng (hộp, túi)
DETECT_MIN_ASPECT = float(os.environ.get('DETECT_MIN_ASPECT', '0.25'))  # v\u1eadt kh\u00f4ng qu\u00e1 g\u1ea7y
DETECT_MAX_ASPECT = float(os.environ.get('DETECT_MAX_ASPECT', '4.00'))  # v\u1eadt kh\u00f4ng qu\u00e1 d\u00e0i

# Edge proximity — nới rộng để bắt vật sát cạnh ảnh (khi chụp gần)
DETECT_EDGE_MARGIN = float(os.environ.get('DETECT_EDGE_MARGIN', '0.02'))

# FIX #7: Center bias giảm penalty — vật thể thường nằm ở rìa khi chụp gần
CENTER_BIAS_SIGMA    = float(os.environ.get('CENTER_BIAS_SIGMA', '0.60'))   # tăng từ 0.45 → bias nhẹ hơn
CENTER_BIAS_MAX_PEN  = float(os.environ.get('CENTER_BIAS_MAX_PEN', '0.20')) # tối đa chỉ phạt 20% thay vì 40%

# FIX #5: Saliency percentile — hạ để bắt vật màu tương đồng background
SALIENCY_PERCENTILE = int(os.environ.get('SALIENCY_PERCENTILE', '75'))  # hạ từ 82

# Confidence tối thiểu để tin vào crop (không refinement lại)
REFINEMENT_THRESHOLD = float(os.environ.get('REFINEMENT_THRESHOLD', '0.55'))  # tăng từ 0.65 → ít gọi refinement hơn

# ============================================================

waste_categories = {
    "recyclable":     ["paper", "plastic", "metal", "glass"],
    "organic":        ["biological"],
    "non_recyclable": ["trash", "shoes"],
    "hazardous":      ["battery"],
    "special":        ["clothes"],
}


def get_recyclability(waste_type):
    for category, items in waste_categories.items():
        if waste_type in items:
            return {
                "recyclable":     "Có thể tái chế",
                "organic":        "Hữu cơ",
                "non_recyclable": "Không tái chế được",
                "hazardous":      "Nguy hại",
                "special":        "Đặc biệt",
            }.get(category, "Không xác định")
    return "Không xác định"


recyclability_descriptions = {
    "Có thể tái chế": "Các loại rác như giấy, nhựa, kim loại và thủy tinh có thể tái chế để sử dụng lại.",
    "Hữu cơ":         "Rác hữu cơ bao gồm thức ăn thừa, rau củ và các vật liệu phân hủy sinh học.",
    "Không tái chế được": "Các loại rác như túi nilon, hộp xốp và giày dép không thể tái chế được.",
    "Nguy hại":       "Pin, hóa chất và các loại rác độc hại cần được xử lý đặc biệt.",
    "Đặc biệt":       "Quần áo cũ và các vật dụng tương tự có thể được quyên góp hoặc tái sử dụng.",
}

waste_types = {
    0: 'battery', 1: 'biological', 2: 'clothes', 3: 'glass',
    4: 'metal',   5: 'paper',     6: 'plastic',  7: 'shoes', 8: 'trash',
}


def load_class_indices_mapping():
    json_path = os.path.join(BASE_DIR, 'class_indices.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                cls = json.load(f)
            # cls: {"0": "battery", ...} → map int(key) → value
            mapping = {int(k): v for k, v in cls.items()}
            logger.info("Loaded class_indices.json with %d entries", len(mapping))
            return mapping
        except Exception as e:
            logger.exception("Failed to load class_indices.json: %s", e)
    logger.warning("Using built-in waste_types mapping (fallback)")
    return dict(waste_types)


waste_types = load_class_indices_mapping()


def validate_model_class_mapping(loaded_model):
    if loaded_model is None:
        return
    try:
        out_shape = loaded_model.output_shape
        if isinstance(out_shape, (list, tuple)) and out_shape:
            out_shape = out_shape[0]
        num_out = int(out_shape[-1]) if out_shape else None
        num_classes = len(waste_types)
        if num_out and num_out != num_classes:
            logger.error("Model có %d lớp nhưng class_indices có %d", num_out, num_classes)
        else:
            logger.info("Ánh xạ lớp OK: %d nhãn %s", num_classes, list(waste_types.values()))
    except Exception as e:
        logger.warning("Không kiểm tra được output_shape model: %s", e)


# ----- DB Models -----
class User(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(255), nullable=False, server_default='')
    points   = db.Column(db.Integer, default=0)
    role     = db.Column(db.String(20), default='user', nullable=False)

    def __init__(self, username=None, password=None, points=0, role='user', **kwargs):
        if username is not None: kwargs['username'] = username
        if password is not None: kwargs['password'] = password
        if points is not None: kwargs['points'] = points
        if role is not None: kwargs['role'] = role
        for key, value in kwargs.items():
            setattr(self, key, value)


class CustomerData(db.Model):
    id      = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    email   = db.Column(db.String(150), nullable=True)
    phone   = db.Column(db.String(20), nullable=True)
    address = db.Column(db.String(255), nullable=True)
    user    = db.relationship("User", backref="customer_data")

    def __init__(self, user_id=None, email=None, phone=None, address=None, **kwargs):
        if user_id is not None: kwargs['user_id'] = user_id
        if email is not None: kwargs['email'] = email
        if phone is not None: kwargs['phone'] = phone
        if address is not None: kwargs['address'] = address
        for key, value in kwargs.items():
            setattr(self, key, value)


class UserActivity(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    user_id   = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    action    = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(500), nullable=True)
    image_url = db.Column(db.String(500), nullable=True)
    image_filename = db.Column(db.String(255), nullable=True)
    user      = db.relationship("User", backref="activities")

    def __init__(self, user_id=None, action=None, timestamp=None,
                 image_path=None, image_url=None, image_filename=None, **kwargs):
        if user_id is not None: kwargs['user_id'] = user_id
        if action is not None: kwargs['action'] = action
        if timestamp is not None: kwargs['timestamp'] = timestamp
        if image_path is not None: kwargs['image_path'] = image_path
        if image_url is not None: kwargs['image_url'] = image_url
        if image_filename is not None: kwargs['image_filename'] = image_filename
        for key, value in kwargs.items():
            setattr(self, key, value)


class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    slug = db.Column(db.String(255), nullable=False, unique=True, index=True)
    summary = db.Column(db.Text, nullable=False)
    content = db.Column(db.Text, nullable=False)
    image = db.Column(db.String(500), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    category_id = db.Column(db.String(50), nullable=False, default='general')
    author = db.Column(db.String(150), nullable=False, default='Ban biên tập')
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    views = db.Column(db.Integer, default=0, nullable=False)
    is_featured = db.Column(db.Boolean, default=False, nullable=False)
    status = db.Column(db.String(20), default='published', nullable=False)
    section = db.Column(db.String(20), nullable=False, default='article')
    sort_order = db.Column(db.Integer, default=0, nullable=False)

    def __init__(self, title=None, slug=None, summary=None, content=None, image=None,
                 category=None, category_id=None, author=None, created_at=None, updated_at=None,
                 views=0, is_featured=False, status='published', section='article', sort_order=0, **kwargs):
        if title is not None: kwargs['title'] = title
        if slug is not None: kwargs['slug'] = slug
        if summary is not None: kwargs['summary'] = summary
        if content is not None: kwargs['content'] = content
        if image is not None: kwargs['image'] = image
        if category is not None: kwargs['category'] = category
        if category_id is not None: kwargs['category_id'] = category_id
        if author is not None: kwargs['author'] = author
        if created_at is not None: kwargs['created_at'] = created_at
        if updated_at is not None: kwargs['updated_at'] = updated_at
        if views is not None: kwargs['views'] = views
        if is_featured is not None: kwargs['is_featured'] = is_featured
        if status is not None: kwargs['status'] = status
        if section is not None: kwargs['section'] = section
        if sort_order is not None: kwargs['sort_order'] = sort_order
        for key, value in kwargs.items():
            setattr(self, key, value)


def ensure_user_columns():
    inspector = inspect(db.engine)
    if 'user' not in inspector.get_table_names():
        return
    columns = {col['name'] for col in inspector.get_columns('user')}
    with db.engine.begin() as conn:
        if 'role' not in columns:
            conn.execute(text("ALTER TABLE user ADD COLUMN role VARCHAR(20) DEFAULT 'user'"))
        if 'password' not in columns:
            conn.execute(text("ALTER TABLE user ADD COLUMN password VARCHAR(255) DEFAULT '' NOT NULL"))
            
    admin_user = User.query.filter_by(username='admin').first()
    if admin_user and admin_user.role != 'admin':
        admin_user.role = 'admin'
        db.session.commit()


def ensure_user_activity_columns():
    inspector = inspect(db.engine)
    if 'user_activity' not in inspector.get_table_names():
        return
    columns = {col['name'] for col in inspector.get_columns('user_activity')}
    with db.engine.begin() as conn:
        if 'image_path' not in columns:
            conn.execute(text("ALTER TABLE user_activity ADD COLUMN image_path VARCHAR(500)"))
        if 'image_url' not in columns:
            conn.execute(text("ALTER TABLE user_activity ADD COLUMN image_url VARCHAR(500)"))
        if 'image_filename' not in columns:
            conn.execute(text("ALTER TABLE user_activity ADD COLUMN image_filename VARCHAR(255)"))


def ensure_article_columns():
    inspector = inspect(db.engine)
    if 'article' not in inspector.get_table_names():
        return
    columns = {col['name'] for col in inspector.get_columns('article')}
    with db.engine.begin() as conn:
        if 'slug' not in columns:
            conn.execute(text("ALTER TABLE article ADD COLUMN slug VARCHAR(255)"))
        if 'summary' not in columns:
            conn.execute(text("ALTER TABLE article ADD COLUMN summary TEXT"))
        if 'content' not in columns:
            conn.execute(text("ALTER TABLE article ADD COLUMN content TEXT"))
        if 'image' not in columns:
            conn.execute(text("ALTER TABLE article ADD COLUMN image VARCHAR(500)"))
        if 'category' not in columns:
            conn.execute(text("ALTER TABLE article ADD COLUMN category VARCHAR(100)"))
        if 'author' not in columns:
            conn.execute(text("ALTER TABLE article ADD COLUMN author VARCHAR(150) DEFAULT 'Ban biên tập'"))
        if 'created_at' not in columns:
            conn.execute(text("ALTER TABLE article ADD COLUMN created_at DATETIME"))
        if 'updated_at' not in columns:
            conn.execute(text("ALTER TABLE article ADD COLUMN updated_at DATETIME"))
        if 'views' not in columns:
            conn.execute(text("ALTER TABLE article ADD COLUMN views INTEGER DEFAULT 0"))
        if 'is_featured' not in columns:
            conn.execute(text("ALTER TABLE article ADD COLUMN is_featured BOOLEAN DEFAULT 0"))
        if 'status' not in columns:
            conn.execute(text("ALTER TABLE article ADD COLUMN status VARCHAR(20) DEFAULT 'published'"))
        if 'source_url' not in columns:
            conn.execute(text("ALTER TABLE article ADD COLUMN source_url VARCHAR(500)"))
        if 'category_id' not in columns:
            conn.execute(text("ALTER TABLE article ADD COLUMN category_id VARCHAR(50) DEFAULT 'all'"))
        if 'section' not in columns:
            conn.execute(text("ALTER TABLE article ADD COLUMN section VARCHAR(20) DEFAULT 'article'"))
        if 'sort_order' not in columns:
            conn.execute(text("ALTER TABLE article ADD COLUMN sort_order INTEGER DEFAULT 0"))


with app.app_context():
    db.create_all()
    ensure_user_columns()
    ensure_user_activity_columns()
    ensure_article_columns()


def slugify_text(value):
    text_value = unicodedata.normalize('NFKD', value or '')
    text_value = ''.join(ch for ch in text_value if not unicodedata.combining(ch))
    text_value = re.sub(r'[^a-zA-Z0-9]+', '-', text_value.lower()).strip('-')
    return text_value or 'bai-viet'


def unique_article_slug(title, article_id=None):
    base_slug = slugify_text(title)
    slug = base_slug
    counter = 2
    while True:
        query = Article.query.filter_by(slug=slug)
        if article_id is not None:
            query = query.filter(Article.id != article_id)
        if query.first() is None:
            return slug
        slug = f"{base_slug}-{counter}"
        counter += 1


def article_image_relative_path(filename):
    return f"uploads/news/{filename}"


def save_article_image_from_upload(file_storage):
    # Luôn tạo tên file mới từ timestamp để tránh tên file dài từ nguồn ngoài
    # (ảnh từ Google Drive, cloud, v.v. có thể có tên >200 ký tự)
    ext = '.jpg'
    original = file_storage.filename or ''
    if '.' in original:
        raw_ext = original.rsplit('.', 1)[-1].lower()[:8]
        if raw_ext in ('jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp'):
            ext = '.' + ('jpg' if raw_ext == 'jpeg' else raw_ext)
    filename = f"news_{datetime.now().strftime('%Y%m%d%H%M%S%f')}{ext}"
    temp_path = os.path.join(NEWS_UPLOAD_FOLDER, filename)
    os.makedirs(NEWS_UPLOAD_FOLDER, exist_ok=True)
    file_storage.save(temp_path)
    return article_image_relative_path(filename)


def article_image_url(image_path):
    if not image_path:
        return url_for('static', filename='uploads/256de136-d5c8-41f2-8044-2f2b523100d21.png')
    if str(image_path).startswith('http'):
        return image_path
    if str(image_path).startswith('/static/'):
        return image_path
    if str(image_path).startswith('static/'):
        return '/' + image_path
    return url_for('static', filename=image_path)


def format_article_datetime(dt):
    if not dt:
        return '—'
    return dt.strftime('%d/%m/%Y %H:%M')


def article_to_dict(article, include_content=False):
    data = {
        'id': article.id,
        'title': article.title,
        'slug': article.slug,
        'summary': article.summary,
        'image_path': article.image,
        'image': article_image_url(article.image),
        'category': article.category,
        'author': article.author,
        'created_at': article.created_at,
        'date': article.created_at.strftime('%d/%m/%Y') if article.created_at else '—',
        'created_at_text': format_article_datetime(article.created_at),
        'updated_at': article.updated_at,
        'updated_at_text': format_article_datetime(article.updated_at),
        'views': article.views or 0,
        'is_featured': bool(article.is_featured),
        'status': article.status,
        'url': url_for('news_detail', slug=article.slug),
        'category_id': slugify_text(article.category),
    }
    if include_content:
        data['content'] = article.content
    return data


def article_icon_for_category(category_name):
    category_name = (category_name or '').lower()
    if 'ai' in category_name or 'công nghệ' in category_name:
        return 'fa-microchip'
    if 'phân loại' in category_name or 'rác' in category_name:
        return 'fa-recycle'
    if 'việt nam' in category_name:
        return 'fa-flag'
    if 'thế giới' in category_name:
        return 'fa-globe'
    if 'kinh tế' in category_name:
        return 'fa-chart-line'
    if 'chính sách' in category_name or 'pháp luật' in category_name:
        return 'fa-scale-balanced'
    return 'fa-folder-open'


def ensure_seed_articles():
    if Article.query.count() > 0:
        return
    try:
        from article_seed_data import build_seed_articles
    except Exception as exc:
        logger.exception('Failed to import article seed data: %s', exc)
        return
    seed_items = build_seed_articles()
    for item in seed_items:
        title = item.get('title', 'Bài viết')
        slug = unique_article_slug(title)
        item['category_id'] = slugify_text(item.get('category', 'article'))
        db.session.add(Article(slug=slug, **item))
    db.session.commit()


with app.app_context():
    ensure_seed_articles()


def get_waste_info():
    json_path = os.path.join(BASE_DIR, 'waste_info.json')
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Error loading waste_info.json: %s", e)
        return {}


waste_info = get_waste_info()

# ----- Model loading -----
model = None


def load_model_safe(model_path):
    global model
    if not os.path.exists(model_path):
        logger.error("Model file does not exist: %s", model_path)
        return None
    try:
        logger.info("Attempting to load model...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"KerasLayer": hub.KerasLayer},
            compile=False,
        )
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        logger.info("Model loaded from %s", model_path)
        validate_model_class_mapping(model)
        return model
    except Exception as e:
        logger.warning("Standard load failed: %s", e)
        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={"KerasLayer": hub.KerasLayer},
                compile=False,
                safe_mode=False,
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
            )
            logger.info("Model loaded (safe_mode=False)")
            validate_model_class_mapping(model)
            return model
        except Exception as e2:
            logger.warning("Safe mode failed: %s", e2)
            try:
                base_model = tf.keras.applications.MobileNetV2(
                    input_shape=(224, 224, 3), include_top=False, weights='imagenet'
                )
                base_model.trainable = False
                model = tf.keras.Sequential([
                    tf.keras.Input(shape=(224, 224, 3)),
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.Dropout(0.45),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Dense(len(waste_types), activation='softmax'),
                ])
                model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'],
                )
                validate_model_class_mapping(model)
                return model
            except Exception as e3:
                logger.exception("All loading methods failed: %s", e3)
                return None


try:
    load_model_safe(MODEL_PATH)
except Exception:
    logger.exception("Initial model load failed.")

DETECTION_MODEL_SOURCE = os.environ.get(
    'DETECTION_MODEL_SOURCE',
    'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2',
)
_detection_model = None


def load_detection_model(source=None):
    global _detection_model
    if _detection_model is not None:
        return _detection_model
    source = source or DETECTION_MODEL_SOURCE
    try:
        logger.info("Loading object detection model from %s", source)
        _detection_model = hub.load(source)
        logger.info("Object detection model loaded")
    except Exception as e:
        logger.exception("Failed to load detection model: %s", e)
        _detection_model = None
    return _detection_model


# ============================================================
# FIX #1: compute_iou() — sửa bug iy2 dùng sai ymin_b
# ============================================================
def compute_iou(box_a, box_b):
    """
    IoU cho định dạng [ymin, xmin, ymax, xmax] (SSD raw format).
    BUG CŨ: iy2 = min(ymax_a, ymin_b) → sai hoàn toàn.
    """
    ymin_a, xmin_a, ymax_a, xmax_a = box_a
    ymin_b, xmin_b, ymax_b, xmax_b = box_b
    ix1 = max(xmin_a, xmin_b)
    iy1 = max(ymin_a, ymin_b)
    ix2 = min(xmax_a, xmax_b)
    iy2 = min(ymax_a, ymax_b)  # ← FIX: ymax_b thay vì ymin_b
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, xmax_a - xmin_a) * max(0.0, ymax_a - ymin_a)
    area_b = max(0.0, xmax_b - xmin_b) * max(0.0, ymax_b - ymin_b)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_iou_xyxy(box_a, box_b):
    """
    IoU cho định dạng [xmin, ymin, xmax, ymax].
    Đây là định dạng bbox lưu trong detections của app này.
    """
    xmin_a, ymin_a, xmax_a, ymax_a = box_a
    xmin_b, ymin_b, xmax_b, ymax_b = box_b
    ix1 = max(xmin_a, xmin_b)
    iy1 = max(ymin_a, ymin_b)
    ix2 = min(xmax_a, xmax_b)
    iy2 = min(ymax_a, ymax_b)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, xmax_a - xmin_a) * max(0.0, ymax_a - ymin_a)
    area_b = max(0.0, xmax_b - xmin_b) * max(0.0, ymax_b - ymin_b)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def suppress_overlapping_detections(detections, iou_threshold=NMS_IOU_THRESHOLD, class_agnostic=False):
    # Chỉ giữ lại box lớn nhất, score cao nhất cho mỗi loại vật thể
    best_per_label = {}
    for det in detections:
        label = det.get('label')
        if not label:
            continue
        area = _box_dimensions(det.get('bbox', [0,0,0,0]))[2]
        score = det.get('score', 0.0)
        if label not in best_per_label:
            best_per_label[label] = det
        else:
            prev = best_per_label[label]
            prev_area = _box_dimensions(prev.get('bbox', [0,0,0,0]))[2]
            prev_score = prev.get('score', 0.0)
            # Ưu tiên score, nếu bằng thì ưu tiên area lớn hơn
            if score > prev_score or (score == prev_score and area > prev_area):
                best_per_label[label] = det

    # Loại bỏ box lồng trong box lớn (nếu có nhiều box cùng label, chỉ giữ 1)
    kept = list(best_per_label.values())
    # Nếu vẫn còn box lồng nhau (do label khác), loại box nhỏ nằm hoàn toàn trong box lớn
    final = []
    for i, det in enumerate(kept):
        bbox_i = det.get('bbox', [0,0,0,0])
        area_i = _box_dimensions(bbox_i)[2]
        is_inner = False
        for j, other in enumerate(kept):
            if i == j:
                continue
            bbox_j = other.get('bbox', [0,0,0,0])
            area_j = _box_dimensions(bbox_j)[2]
            # Nếu box i nằm hoàn toàn trong box j và nhỏ hơn đáng kể
            if (bbox_i[0] >= bbox_j[0] and bbox_i[1] >= bbox_j[1] and
                bbox_i[2] <= bbox_j[2] and bbox_i[3] <= bbox_j[3] and area_i < area_j * 0.85):
                is_inner = True
                break
        if not is_inner:
            final.append(det)
    return final


def nms_detections(detections, iou_threshold=NMS_IOU_THRESHOLD):
    """Class-agnostic NMS."""
    return suppress_overlapping_detections(detections, iou_threshold=iou_threshold, class_agnostic=True)


def _box_dimensions(bbox):
    """Trả về (w, h, area, aspect_ratio) từ bbox [xmin,ymin,xmax,ymax]."""
    coords = [float(v) for v in bbox]
    xmin, ymin, xmax, ymax = coords
    w = abs(xmax - xmin)
    h = abs(ymax - ymin)
    area = w * h
    aspect = w / h if h > 1e-6 else 0.0
    return w, h, area, aspect


def is_edge_box(bbox, margin=DETECT_EDGE_MARGIN):
    """
    Trả về True nếu box tràn cả 2 chiều (ngang VÀ dọc) — thường là box nền toàn ảnh.
    FIX: margin nhỏ hơn (0.02) để không lọc nhầm vật sát cạnh hợp lệ.
    Điều kiện: phải tràn cả ngang lẫn dọc, không chỉ 1 chiều.
    """
    xmin, ymin, xmax, ymax = [float(v) for v in bbox]
    spans_horizontal = (xmin <= margin) and (xmax >= 1.0 - margin)
    spans_vertical   = (ymin <= margin) and (ymax >= 1.0 - margin)
    # Chỉ loại khi tràn CẢ HAI chiều (box nền thực sự)
    return spans_horizontal and spans_vertical


def filter_detection_boxes(detections):
    """
    Lọc bỏ box bất hợp lệ về:
    - Diện tích (quá nhỏ/lớn)
    - Aspect ratio bất thường
    - Box tràn biên toàn ảnh
    - Label score thấp
    FIX #9: Dùng 1 bộ ngưỡng nhất quán, không check area 2 lần.
    """
    filtered = []
    debug_filtered = []
    for det in detections:
        bbox = det.get('bbox', [0, 0, 0, 0])
        w, h, area, aspect = _box_dimensions(bbox)

        reason = None
        if area < DETECT_MIN_BOX_AREA or area > DETECT_MAX_BOX_AREA:
            reason = f"area={area:.3f} (outside [{DETECT_MIN_BOX_AREA:.3f}, {DETECT_MAX_BOX_AREA:.3f}])"
        elif aspect < DETECT_MIN_ASPECT or aspect > DETECT_MAX_ASPECT:
            reason = f"aspect={aspect:.2f} (outside [{DETECT_MIN_ASPECT:.2f}, {DETECT_MAX_ASPECT:.2f}])"
        elif is_edge_box(bbox):
            reason = f"edge box {bbox}"
        else:
            score = float(det.get('label_score', det.get('score', 0.0)))
            if score < DETECT_MIN_LABEL_SCORE:
                reason = f"label_score={score:.3f} < {DETECT_MIN_LABEL_SCORE:.3f}"

        if reason:
            debug_filtered.append({**det, 'filter_reason': reason})
            continue
        filtered.append(det)

    # Debug: log filtered-out detections for admin review
    if debug_filtered:
        logger.info("Filtered out %d detections:", len(debug_filtered))
        for d in debug_filtered:
            logger.info("Filtered: %s", d)
    return filtered


# ----- Image helpers -----
def get_current_user():
    user_id = session.get('user_id')
    if not user_id:
        return None
    return db.session.get(User, user_id)


def is_admin_user(user):
    if not user:
        return False
    return getattr(user, 'role', 'user') == 'admin' or user.username == 'admin'


def admin_home_url():
    return '/admin/dashboard'


POINTS_PER_OBJECT = 10


def points_for_classification(object_count=1):
    return max(int(object_count), 1) * POINTS_PER_OBJECT


def parse_multi_object_count(action):
    match = re.search(r'Phân loại\s+(\d+)\s+đối tượng', action)
    return int(match.group(1)) if match else None


def trim_bottom_watermark(img):
    img = img.convert('RGB')
    w, h = img.size
    if h < 120 or TRIM_BOTTOM_RATIO <= 0:
        return img
    return img.crop((0, 0, w, max(1, int(h * (1.0 - TRIM_BOTTOM_RATIO)))))


def resize_for_model(img):
    return img.convert('RGB').resize((IMG_CLASSIFY_SIZE, IMG_CLASSIFY_SIZE), Image.LANCZOS)


def square_center_resize_for_model(img, center_ratio=1.0):
    img = img.convert('RGB')
    w, h = img.size
    side = int(min(w, h) * max(0.35, min(center_ratio, 1.0)))
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize(
        (IMG_CLASSIFY_SIZE, IMG_CLASSIFY_SIZE), Image.LANCZOS
    )


def prepare_image_for_classification(img):
    return trim_bottom_watermark(img)


# ============================================================
# FIX #3: TTA — cân bằng lại weight
# - Không ưu tiên full-image quá mức (background bias)
# - Thêm random shift crop để bắt vật ở rìa
# - focus_object=True: KHÔNG crop center nhỏ (FIX #10)
# ============================================================
def _shift_crop(img, shift_x=0.0, shift_y=0.0, ratio=0.85):
    """Crop lệch tâm để bắt vật thể ở rìa."""
    img = img.convert('RGB')
    w, h = img.size
    side = int(min(w, h) * ratio)
    cx = int(w * (0.5 + shift_x))
    cy = int(h * (0.5 + shift_y))
    left  = max(0, min(w - side, cx - side // 2))
    top   = max(0, min(h - side, cy - side // 2))
    return img.crop((left, top, left + side, top + side)).resize(
        (IMG_CLASSIFY_SIZE, IMG_CLASSIFY_SIZE), Image.LANCZOS
    )


def build_tta_variants(img, focus_object=False):
    """
    TTA mở rộng: nhiều kiểu crop/lật để tăng robust.
    """
    img = prepare_image_for_classification(img)
    variants = [
        (resize_for_model(img),                      1.20),
        (square_center_resize_for_model(img, 0.95),  1.10),
        (square_center_resize_for_model(img, 0.90),  1.20),
        (square_center_resize_for_model(img, 0.80),  1.10),
        (square_center_resize_for_model(img, 0.70),  0.90),
    ]
    if CLASSIFICATION_USE_TTA:
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        variants.append((resize_for_model(flipped), 0.80))
        variants.append((square_center_resize_for_model(flipped, 0.90), 0.80))
        if not focus_object:
            # Shift crop: bắt vật thể lệch khỏi tâm (phổ biến khi chụp điện thoại)
            for sx, sy, w in [
                ( 0.15,  0.15, 0.90), ( -0.15,  0.15, 0.90),
                ( 0.15, -0.15, 0.80), ( -0.15, -0.15, 0.80),
                ( 0.22,  0.00, 0.70), ( 0.00,  0.22, 0.70),
            ]:
                variants.append((_shift_crop(img, sx, sy), w))
    return variants


def _class_index(label):
    for idx, name in waste_types.items():
        if name == label:
            return int(idx)
    return None


def _top_labels_from_probs(probs, n=3):
    ordered = np.argsort(probs)[::-1]
    return [
        (waste_types.get(int(i)), float(probs[int(i)]))
        for i in ordered[:n]
        if waste_types.get(int(i))
    ]


def square_center_region(img, center_ratio=0.65):
    img = img.convert('RGB')
    w, h = img.size
    side = int(min(w, h) * max(0.35, min(center_ratio, 1.0)))
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _glass_vs_organic_visual_score(pil_img):
    crop = square_center_region(pil_img, 0.58)
    arr  = np.array(crop).astype(np.float32)
    gray = arr.mean(axis=2)
    edge_strength = float(np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean())
    specular  = float((gray > 195).mean())
    r, g, b   = arr[..., 0], arr[..., 1], arr[..., 2]
    warm_glass = float(((r > g) & (g >= b * 0.82) & (r > 70) & (r < 215)).mean())
    return min(
        1.0,
        0.4 * min(edge_strength / 22.0, 1.0)
        + 0.35 * min(specular * 10.0, 1.0)
        + 0.25 * min(warm_glass * 2.5, 1.0),
    )


def disambiguate_glass_biological(img, probs):
    if probs is None or probs.size == 0:
        return probs
    bio_idx   = _class_index('biological')
    glass_idx = _class_index('glass')
    if bio_idx is None or glass_idx is None:
        return probs
    top = _top_labels_from_probs(probs, 4)
    if not top:
        return probs
    best_label, best_score = top[0]
    glass_score = float(probs[glass_idx])
    bio_score   = float(probs[bio_idx])
    top_names   = [name for name, _ in top]

    if glass_score < 0.10 or 'glass' not in top_names[:3]:
        return probs
    visual = _glass_vs_organic_visual_score(img)
    if visual < 0.30:
        return probs
    margin = bio_score - glass_score
    if not (best_label == 'biological' and best_score < 0.70 and margin < 0.40 and visual >= 0.35):
        return probs

    adjusted = probs.copy()
    boost = 0.06 + 0.18 * visual + max(0.0, 0.10 - margin * 0.12)
    adjusted[glass_idx] = glass_score + boost
    adjusted[bio_idx]   = max(0.0, bio_score - boost * 0.9)
    total = adjusted.sum()
    if total > 0:
        adjusted /= total
    logger.info("Glass/biological disambiguate: visual=%.2f margin=%.2f", visual, margin)
    return adjusted


# ============================================================
# FIX #4: refine_low_confidence_probs — cải thiện refinement
# - Ngưỡng kích hoạt tăng lên 0.55 (ít gọi hơn)
# - Thay vì focus_object=True (crop center → miss vật rìa),
#   thử nhiều scale crop rồi lấy kết quả tốt nhất
# - Merge theo confidence thực tế, không cứng 30/70
# ============================================================
def _predict_multi_scale(img):
    """
    Predict ở nhiều scale crop, trả về probs tốt nhất.
    Dùng thay cho predict_probabilities(focus_object=True).
    """
    img = prepare_image_for_classification(img)
    w, h = img.size

    # Thử các vùng: center, 4 góc, full
    crops = [
        resize_for_model(img),  # full
        square_center_resize_for_model(img, 0.85),
        _shift_crop(img,  0.12,  0.12, 0.80),
        _shift_crop(img, -0.12,  0.12, 0.80),
        _shift_crop(img,  0.12, -0.12, 0.80),
        _shift_crop(img, -0.12, -0.12, 0.80),
    ]
    batch = preprocess_batch(crops)
    preds = run_model_on_batch(batch)
    if preds is None:
        return None

    # Trả về crop có confidence cao nhất
    best_idx = int(np.argmax(np.max(preds, axis=1)))
    return preds[best_idx]


def refine_low_confidence_probs(img, probs):
    """
    FIX #4: Chỉ kích hoạt khi confidence thực sự thấp.
    Thay vì focus center crop (miss vật rìa), thử multi-scale crop.
    """
    if probs is None:
        return probs
    best_score = float(np.max(probs))
    if best_score >= REFINEMENT_THRESHOLD:
        return probs
    logger.info("Confidence thấp (%.2f < %.2f), chạy multi-scale refinement...", best_score, REFINEMENT_THRESHOLD)
    refined = _predict_multi_scale(img)
    if refined is None:
        return probs
    refined_score = float(np.max(refined))
    if refined_score <= best_score:
        logger.info("Multi-scale không cải thiện (%.2f ≤ %.2f), giữ probs gốc", refined_score, best_score)
        return probs
    # Merge theo tỉ lệ dynamic: refined tốt hơn → weight cao hơn
    w_orig    = 1.0 - min(0.80, (refined_score - best_score) * 2.0)
    w_refined = 1.0 - w_orig
    merged = w_orig * probs + w_refined * refined
    total  = merged.sum()
    merged = merged / total if total > 0 else probs
    logger.info("Refined: %.2f → %.2f (w_orig=%.2f)", best_score, float(np.max(merged)), w_orig)
    return merged


def preprocess_batch(pil_images):
    tensors = [
        tf.keras.applications.mobilenet_v2.preprocess_input(
            np.array(pil_img).astype(np.float32)
        )
        for pil_img in pil_images
    ]
    return np.stack(tensors, axis=0)


def run_model_on_batch(batch):
    global model
    if model is None:
        return None
    try:
        out   = model(batch, training=False)
        preds = out.numpy() if hasattr(out, 'numpy') else np.asarray(out)
    except Exception:
        inputs = getattr(model, 'inputs', None)
        if isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
            preds = model.predict([batch, batch], verbose=0)
        else:
            preds = model.predict(batch, verbose=0)
        preds = np.asarray(preds)
    if preds.ndim == 2:
        return preds
    if preds.ndim == 1:
        return np.expand_dims(preds, axis=0)
    return preds.reshape(1, -1)


def predict_probabilities(img, focus_object=False):
    variants = build_tta_variants(img, focus_object=focus_object)
    pil_list = [v[0] for v in variants]
    weights  = np.asarray([v[1] for v in variants], dtype=np.float32)
    batch    = preprocess_batch(pil_list)
    preds    = run_model_on_batch(batch)
    if preds is None:
        return None
    weights = weights[: preds.shape[0]]
    weights = weights / weights.sum()
    return np.average(preds, axis=0, weights=weights)


def resolve_label_from_probs(probs):
    if probs is None or probs.size == 0:
        return None, 0.0
    ordered    = np.argsort(probs)[::-1]
    best_idx   = int(ordered[0])
    best_score = float(probs[best_idx])
    label      = waste_types.get(best_idx, 'trash')
    top3       = [(waste_types.get(int(i)), float(probs[int(i)])) for i in ordered[:3] if waste_types.get(int(i))]
    logger.info("Top dự đoán: %s → %s %.2f", top3, label, best_score)
    return label, best_score


def classify_pil_image_with_score(img):
    global model
    if model is None:
        logger.error("Model not loaded.")
        return None, 0.0
    try:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        # Đảm bảo resize đúng 224x224 và chuẩn hóa input
        img = prepare_image_for_classification(img)
        img_resized = img.resize((224, 224), Image.LANCZOS)
        arr = np.array(img_resized).astype(np.float32)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        batch = np.expand_dims(arr, axis=0)
        # Predict với TTA nếu bật, nếu không thì predict trực tiếp
        if CLASSIFICATION_USE_TTA:
            probs = predict_probabilities(img)
        else:
            preds = run_model_on_batch(batch)
            probs = preds[0] if preds is not None else None
        if probs is None:
            return None, 0.0
        # FIX #4: refinement cải tiến
        probs = refine_low_confidence_probs(img, probs)
        probs = disambiguate_glass_biological(img, probs)
        label, score = resolve_label_from_probs(probs)

        # Nếu score vẫn thấp, thử full-image không crop/trim
        if score < SINGLE_MIN_CONFIDENCE:
            logger.info("Score %.2f < %.2f, thử full-image fallback...", score, SINGLE_MIN_CONFIDENCE)
            arr_fb = np.array(img.resize((224, 224), Image.LANCZOS)).astype(np.float32)
            arr_fb = tf.keras.applications.mobilenet_v2.preprocess_input(arr_fb)
            batch_fb = np.expand_dims(arr_fb, axis=0)
            preds_fb = run_model_on_batch(batch_fb)
            if preds_fb is not None:
                fb_probs = preds_fb[0]
                fb_label, fb_score = resolve_label_from_probs(fb_probs)
                if fb_score > score:
                    logger.info("Full-image fallback tốt hơn: %s %.2f > %.2f", fb_label, fb_score, score)
                    label, score = fb_label, fb_score

        return label, score
    except Exception as e:
        logger.exception("classify_pil_image_with_score error: %s", e)
        return None, 0.0


def classify_crop_robust(pil_crop):
    """
    Classify một crop (bbox từ SSD/saliency).
    Dùng focus_object=True: không thêm shift crop (đã là crop vật thể).
    """
    return classify_pil_image_with_score(pil_crop)


def classify_pil_image(img):
    label, _ = classify_pil_image_with_score(img)
    return label


def load_rgb_image(image_path):
    try:
        with Image.open(image_path) as im:
            im.load()
            return im.convert('RGB')
    except Exception as e:
        raise ValueError("Không đọc được file ảnh.") from e


def ensure_model_ready():
    global model
    if model is None:
        load_model_safe(MODEL_PATH)
    if model is None:
        return False, "Mô hình AI chưa được tải."
    return True, None


def safe_upload_filename(original_name, prefix='upload'):
    ext = '.jpg'
    if original_name and '.' in original_name:
        raw_ext = original_name.rsplit('.', 1)[-1].lower()[:8]
        if raw_ext in ('jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp'):
            ext = '.' + ('jpg' if raw_ext == 'jpeg' else raw_ext)
    name = secure_filename(original_name or '') or ''
    # Giới hạn tên file tối đa 80 ký tự (tránh lỗi MAX_PATH trên Windows)
    if name and name not in ('.', '..'):
        stem = name[:80] if len(name) > 80 else name
        # Đảm bảo phần extension đúng sau khi cắt
        if '.' in stem:
            stem = stem.rsplit('.', 1)[0]
        return f"{stem[:60]}{ext}"
    return f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}{ext}"


def save_image_from_upload(file_storage):
    filename  = safe_upload_filename(file_storage.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_storage.save(temp_path)
    dest_path = os.path.join(STATIC_UPLOAD_FOLDER, filename)
    if not os.path.exists(dest_path):
        shutil.move(temp_path, dest_path)
    else:
        os.remove(temp_path)
    return dest_path, filename


def save_image_from_base64(captured_image):
    raw = (captured_image or '').strip()
    if not raw:
        return None, None
    _, b64 = raw.split(',', 1) if ',' in raw else ('', raw)
    try:
        image_data = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise ValueError("Dữ liệu ảnh không hợp lệ.") from e
    if len(image_data) < 32:
        raise ValueError("Ảnh quá nhỏ hoặc rỗng.")
    filename  = f"captured_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
    dest_path = os.path.join(STATIC_UPLOAD_FOLDER, filename)
    with open(dest_path, 'wb') as f:
        f.write(image_data)
    return dest_path, filename


def resolve_uploaded_image_from_request():
    captured_image = request.form.get('captured_image', '').strip()
    if captured_image:
        return save_image_from_base64(captured_image)
    file = request.files.get('file')
    if file and file.filename:
        return save_image_from_upload(file)
    return None, None


def run_classification_for_image(dest_path, filename, user, mode, is_admin):
    ok, _ = ensure_model_ready()
    if not ok:
        logger.error("Model chưa sẵn sàng")
        return None
    try:
        load_rgb_image(dest_path)
    except ValueError as e:
        logger.warning("Ảnh không hợp lệ: %s", e)
        return None

    img    = load_rgb_image(dest_path)
    result, confidence = classify_pil_image_with_score(img)

    if mode == 'multi':
        try:
            detections = detect_objects(dest_path)
        except Exception as e:
            logger.exception("detect_objects failed: %s", e)
            detections = []

        # Không detect được → phân loại cả ảnh
        if not detections and result:
            return _render_single_from_result(result, confidence, dest_path, filename, user, is_admin)

        # FIX #8: So sánh score detect vs full-image, không luôn fallback
        if len(detections) == 1:
            det_score = float(detections[0].get('label_score', 0))
            # Chỉ fallback khi detect YẾU và full-image MẠNH HƠN ĐÁng kể
            if det_score < SINGLE_DET_FALLBACK_SCORE and confidence > det_score + 0.15:
                logger.info(
                    "Single weak detection (%.2f), full-image tốt hơn hẳn (%.2f), fallback",
                    det_score, confidence
                )
                return _render_single_from_result(result, confidence, dest_path, filename, user, is_admin)
            # Nếu detect tạm được, vẫn dùng detection thay vì fallback
            logger.info("Single detection OK (%.2f), dùng detection result", det_score)

        if detections:
            points_earned = points_for_classification(len(detections))
            user.points  += points_earned
            image_payload = build_activity_image_payload(dest_path, filename)
            # Ghi log tổng hợp cho đa vật thể — chỉ một bản ghi duy nhất
            db.session.add(UserActivity(user_id=user.id, action=f"Phân loại {len(detections)} đối tượng", **image_payload))
            db.session.commit()
            return build_multi_object_response(dest_path, filename, detections, is_admin)

    if result:
        return _render_single_from_result(result, confidence, dest_path, filename, user, is_admin)
    return None


def _render_single_from_result(result, confidence, dest_path, filename, user, is_admin):
    user.points += points_for_classification(1)
    db.session.commit()
    db.session.add(UserActivity(user_id=user.id, action=f"Phân loại: {result}", **build_activity_image_payload(dest_path, filename)))
    db.session.commit()
    return render_single_classification_result(result, dest_path, filename, is_admin, confidence_score=confidence)


def classify_image_with_score(image_path):
    try:
        img = load_rgb_image(image_path)
        return classify_pil_image_with_score(img)
    except ValueError:
        raise
    except Exception as e:
        logger.exception("classify_image_with_score error: %s", e)
        return None, 0.0


def classify_image(image_path):
    label, _ = classify_image_with_score(image_path)
    return label


WASTE_BOX_COLORS = {
    'plastic': '#2563eb', 'glass':  '#0891b2', 'paper':    '#ca8a04',
    'metal':   '#64748b', 'battery':'#dc2626', 'biological':'#16a34a',
    'clothes': '#7c3aed', 'shoes':  '#c2410c', 'trash':     '#6b7280',
}


def annotate_detections(image_path, detections, filename):
    img   = Image.open(image_path).convert("RGB")
    draw  = ImageDraw.Draw(img)
    width, height = img.size
    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except Exception:
        font = None

    for det in detections:
        xmin, ymin, xmax, ymax = det['bbox']
        left   = max(0,    min(width  - 2, int(xmin * width)))
        top    = max(0,    min(height - 2, int(ymin * height)))
        right  = max(left + 2, min(width,  int(xmax * width)))
        bottom = max(top  + 2, min(height, int(ymax * height)))
        label  = det.get('label', 'unknown')
        score  = float(det.get('label_score', det.get('score', 0.0)))
        color  = WASTE_BOX_COLORS.get(label, '#dc2626')
        draw.rectangle([left, top, right, bottom], outline=color, width=4)
        text = f"{label} {score:.2f}"
        try:
            if hasattr(draw, 'textbbox'):
                bbox = draw.textbbox((0, 0), text, font=font)
                text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            elif font and hasattr(font, 'getsize'):
                text_size = font.getsize(text)
            elif hasattr(draw, 'textsize'):
                text_size = draw.textsize(text, font=font)
            else:
                text_size = (len(text) * 6, 12)
        except Exception:
            text_size = (len(text) * 6, 12)
        bg = [left, max(top - text_size[1] - 4, 0), left + text_size[0] + 8, top]
        draw.rectangle(bg, fill=color)
        draw.text((left + 3, max(top - text_size[1] - 2, 0)), text, fill="white", font=font)

    annotated_name = f"annotated_{filename}"
    img.save(os.path.join(STATIC_UPLOAD_FOLDER, annotated_name))
    return annotated_name


def upscale_crop_if_needed(crop, min_side=224):
    if min(crop.size) >= min_side:
        return crop
    scale = min_side / min(crop.size)
    return crop.resize(
        (max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
        Image.LANCZOS,
    )


# ============================================================
# FIX #7: center_bias_score — giảm penalty vật ở rìa
# Người dùng thường đặt vật ở góc dưới khi chụp điện thoại
# sigma lớn hơn → bias nhẹ hơn, max penalty 20% thay vì 40%
# ============================================================
def center_bias_score(bbox, sigma=CENTER_BIAS_SIGMA, max_penalty=CENTER_BIAS_MAX_PEN):
    """
    bbox: [xmin, ymin, xmax, ymax] normalized [0,1].
    FIX: sigma=0.60 (tăng từ 0.45), max_penalty=0.20 (giảm từ 0.40).
    Vật ở rìa chỉ bị phạt tối đa 20%, không bị loại sai.
    """
    xmin, ymin, xmax, ymax = [float(v) for v in bbox]
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
    raw_score = float(np.exp(-0.5 * (dist / sigma) ** 2))
    # Clamp: tối thiểu 1 - max_penalty (không phạt quá mức)
    return max(1.0 - max_penalty, raw_score)


# ============================================================
# FIX #2 + #5 + #11: SALIENCY DETECTION — cải tiến toàn diện
#
# FIX #2: Border median dùng multi-sample margin, không bị nhiễm
#         bởi vật thể sát cạnh (lấy median của 3 margin width khác nhau)
# FIX #5: Percentile adaptive theo độ tương phản ảnh
# FIX #11: Adaptive threshold theo histogram ảnh
# FIX #12: Multi-scale border margin
# ============================================================
def _compute_robust_background(small_f, sh, sw):
    """
    FIX #2 + #12: Tính background median dùng multi-scale margin.
    Lấy giao của 3 margin size → giảm nguy cơ vật sát cạnh bị tính là background.
    """
    results = []
    for frac in (14, 18, 22):  # 3 margin size khác nhau
        m = max(3, min(sh, sw) // frac)
        border = np.concatenate([
            small_f[:m, :].reshape(-1, 3),
            small_f[-m:, :].reshape(-1, 3),
            small_f[:, :m].reshape(-1, 3),
            small_f[:, -m:].reshape(-1, 3),
        ])
        results.append(np.median(border, axis=0))
    # Median của 3 margin → ổn định hơn
    return np.median(results, axis=0)


def _adaptive_saliency_percentile(dist_map):
    """
    FIX #5 + #11: Chọn percentile động theo độ tương phản ảnh.
    - Ảnh tương phản thấp (vật màu giống background): percentile thấp hơn (65-70)
    - Ảnh tương phản cao: percentile 78
    """
    p25, p75 = float(np.percentile(dist_map, 25)), float(np.percentile(dist_map, 75))
    contrast_iqr = p75 - p25  # interquartile range = proxy for contrast

    if contrast_iqr < 15.0:
        # Ảnh tương phản thấp → hạ ngưỡng mạnh
        percentile = 65
    elif contrast_iqr < 30.0:
        percentile = 70
    elif contrast_iqr < 50.0:
        percentile = 75
    else:
        percentile = SALIENCY_PERCENTILE  # mặc định 75

    logger.debug("Saliency adaptive percentile=%d (contrast_iqr=%.1f)", percentile, contrast_iqr)
    return percentile


def detect_objects_saliency(image_path, max_proposals=None):
    if not USE_SALIENCY_DETECTION:
        return []
    max_proposals = max_proposals or MAX_MULTI_DETECTIONS
    try:
        from scipy import ndimage
    except ImportError:
        logger.warning("scipy không có — bỏ qua saliency detection")
        return []
    try:
        img_pil = prepare_image_for_classification(load_rgb_image(image_path))
        img     = np.array(img_pil)
        h, w    = img.shape[:2]
        max_dim = 560
        scale   = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            sw, sh = int(w * scale), int(h * scale)
            small  = np.array(Image.fromarray(img).resize((sw, sh), Image.LANCZOS))
        else:
            small = img; sh, sw = h, w; scale = 1.0

        small_f = small.astype(np.float32)

        # FIX #2 + #12: robust multi-scale background
        bg   = _compute_robust_background(small_f, sh, sw)
        dist = np.linalg.norm(small_f - bg, axis=2)

        # FIX #5 + #11: adaptive percentile
        percentile = _adaptive_saliency_percentile(dist)
        thresh = max(15.0, float(np.percentile(dist, percentile)))
        mask   = dist > thresh

        # Morphology — giữ cấu trúc nhưng ít aggressive hơn
        mask = ndimage.binary_opening( mask, structure=np.ones((5, 5), dtype=bool))
        mask = ndimage.binary_closing( mask, structure=np.ones((9, 9), dtype=bool))
        mask = ndimage.binary_fill_holes(mask)

        labeled, num_features = ndimage.label(mask)
        min_area = sh * sw * SALIENCY_MIN_AREA_RATIO
        max_area = sh * sw * SALIENCY_MAX_AREA_RATIO
        detections = []

        for label_id in range(1, num_features + 1):
            component = labeled == label_id
            area      = float(component.sum())
            if area < min_area or area > max_area:
                continue
            rows = np.any(component, axis=1)
            cols = np.any(component, axis=0)
            if not rows.any() or not cols.any():
                continue
            rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
            cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])

            rect_h = rmax - rmin + 1
            rect_w = cmax - cmin + 1
            aspect = rect_w / rect_h if rect_h > 0 else 0.0
            if aspect < DETECT_MIN_ASPECT or aspect > DETECT_MAX_ASPECT:
                logger.debug("Saliency: loại vùng aspect=%.2f", aspect)
                continue

            pad     = int(0.025 * max(rect_h, rect_w))  # padding nhỏ hơn
            left_s  = max(0,  cmin - pad); top_s   = max(0,  rmin - pad)
            right_s = min(sw, cmax + 1 + pad); bottom_s = min(sh, rmax + 1 + pad)

            left   = int(left_s  / scale); top    = int(top_s   / scale)
            right  = int(right_s / scale); bottom = int(bottom_s / scale)
            if right - left < 48 or bottom - top < 48:
                continue

            norm_bbox = [left / w, top / h, right / w, bottom / h]

            # FIX: chỉ loại box tràn cả 2 chiều
            if is_edge_box(norm_bbox):
                logger.debug("Saliency: loại box tràn biên toàn ảnh")
                continue

            crop = upscale_crop_if_needed(img_pil.crop((left, top, right, bottom)))
            label, label_score = classify_crop_robust(crop)
            if not label or label_score < SALIENCY_MIN_LABEL_SCORE:
                continue

            # FIX #7: center bias nhẹ hơn
            cb = center_bias_score(norm_bbox)
            effective_score = label_score * cb  # max penalty 20%

            detections.append({
                'bbox':        norm_bbox,
                'score':       effective_score,
                'label':       label,
                'label_score': label_score,
                'center_bias': cb,
                'source':      'saliency',
            })

        detections.sort(key=lambda d: d.get('score', 0.0), reverse=True)
        detections = nms_detections(detections[:max_proposals * 2])  # NMS trên pool rộng hơn
        detections = detections[:max_proposals]
        logger.info("Saliency: %d vật thể (percentile=%d)", len(detections), percentile)
        return detections
    except Exception as e:
        logger.exception("Saliency detection failed: %s", e)
        return []


def detect_objects_grid_fallback(image_path, grid_cols=3, grid_rows=3):
    if not GRID_FALLBACK_ENABLED:
        return []
    try:
        img   = Image.open(image_path).convert('RGB')
        w, h  = img.size
        detections = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                left   = int(col * w / grid_cols);   top    = int(row * h / grid_rows)
                right  = int((col+1) * w / grid_cols); bottom = int((row+1) * h / grid_rows)
                if right - left < 48 or bottom - top < 48:
                    continue
                crop  = upscale_crop_if_needed(img.crop((left, top, right, bottom)))
                label, label_score = classify_crop_robust(crop)
                if not label or label_score < DETECT_GRID_MIN_SCORE:
                    continue
                detections.append({
                    'bbox': [left/w, top/h, right/w, bottom/h],
                    'score': label_score, 'label': label, 'label_score': label_score,
                })
        if detections:
            detections = suppress_overlapping_detections(detections, iou_threshold=0.30)
        return detections
    except Exception as e:
        logger.exception("Grid fallback failed: %s", e)
        return []


# ============================================================
# FIX #6: SSD DETECTION — post-process box sau khi classify
# SSD đề xuất boxes theo COCO (80 class), không phải rác.
# Cải tiến: expand box nhẹ để bắt đủ vật thể, rồi dùng
# classifier làm "verifier" chính thay vì SSD score.
# ============================================================
def _expand_ssd_box(ymin, xmin, ymax, xmax, width, height, pad_ratio=0.04):
    """
    Mở rộng box SSD thêm pad_ratio để bắt đủ vật thể.
    SSD thường crop sát biên vật thể → mất texture.
    """
    bw = xmax - xmin
    bh = ymax - ymin
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio
    left   = max(0,      int((xmin - pad_x) * width))
    top    = max(0,      int((ymin - pad_y) * height))
    right  = min(width,  int((xmax + pad_x) * width))
    bottom = min(height, int((ymax + pad_y) * height))
    return left, top, right, bottom


def detect_objects_ssd(
    image_path,
    min_score=DETECT_MIN_SCORE,
    max_detections=100,
    min_label_score=DETECT_MIN_LABEL_SCORE,
    min_box_area=DETECT_MIN_BOX_AREA,
    max_box_area=DETECT_MAX_BOX_AREA,
):
    detection_model = load_detection_model()
    if detection_model is None:
        return []

    img    = Image.open(image_path).convert("RGB")
    width, height = img.size
    arr    = np.array(img)
    inp    = tf.convert_to_tensor(arr, dtype=tf.uint8)[tf.newaxis, ...]
    sig    = detection_model.signatures['serving_default']
    out    = sig(inp)

    boxes  = out['detection_boxes'][0].numpy()
    scores = out['detection_scores'][0].numpy()
    num    = int(out['num_detections'][0].numpy())

    min_pixel = 56  # hạ từ 72: bắt vật nhỏ hơn
    detections = []
    for i in range(min(max_detections, num)):
        score = float(scores[i])
        if score < min_score:
            continue
        ymin, xmin, ymax, xmax = boxes[i].tolist()
        box_area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
        if box_area < min_box_area or box_area > max_box_area:
            continue

        bw = max(0.0, xmax - xmin)
        bh = max(0.0, ymax - ymin)
        if bh < 1e-6:
            continue
        aspect = bw / bh
        if aspect < DETECT_MIN_ASPECT or aspect > DETECT_MAX_ASPECT:
            continue

        # FIX: chỉ loại box tràn cả 2 chiều
        norm_bbox_check = [xmin, ymin, xmax, ymax]
        if is_edge_box(norm_bbox_check):
            logger.debug("SSD: loại box tràn biên toàn ảnh")
            continue

        # FIX #6: expand box để bắt đủ vật thể
        left, top, right, bottom = _expand_ssd_box(ymin, xmin, ymax, xmax, width, height)

        if (right - left) < min_pixel or (bottom - top) < min_pixel:
            continue
        if right <= left or bottom <= top:
            continue

        crop  = upscale_crop_if_needed(img.crop((left, top, right, bottom)))
        label, label_score = classify_crop_robust(crop)
        combined = score * label_score
        if not label or label_score < min_label_score or combined < DETECT_MIN_COMBINED_SCORE:
            continue

        norm_bbox = [left/width, top/height, right/width, bottom/height]

        # FIX #7: center bias nhẹ hơn
        cb = center_bias_score(norm_bbox)
        effective_score = label_score * cb  # max penalty 20%

        detections.append({
            'bbox':        norm_bbox,
            'score':       effective_score,
            'label':       label,
            'label_score': label_score,
            'center_bias': cb,
            'source':      'ssd',
        })

    if detections:
        detections = nms_detections(detections)
    detections.sort(key=lambda d: d.get('score', 0.0), reverse=True)
    return detections[:MAX_MULTI_DETECTIONS]


# ============================================================
# FIX #9: filter_background_detections — nhất quán với filter_detection_boxes
# Bỏ kiểm tra area lần 2 (đã check trong filter_detection_boxes).
# Chỉ lọc box thực sự là background toàn ảnh.
# ============================================================
def filter_background_detections(detections):
    """
    FIX #9: Chỉ lọc background thực sự, không check area 2 lần.
    - Box tràn cả 2 chiều và area > 0.80 → background
    - corner_bias < 0.20 VÀ score rất thấp → background corner
    """
    if not detections:
        return detections
    filtered = []
    for det in detections:
        bbox = det.get('bbox', [0, 0, 1, 1])
        _, _, area, _ = _box_dimensions(bbox)
        cb = det.get('center_bias', center_bias_score(bbox))
        score = det.get('score', 0.0)
        label_score = det.get('label_score', score)

        # Bỏ box gần như toàn ảnh (area > 80%) — chắc chắn là background
        if area > 0.80:
            logger.debug("filter_background: loại box area=%.3f (>0.80)", area)
            continue

        # Bỏ box ở góc với confidence rất thấp
        # FIX: threshold thấp hơn (0.15 thay vì 0.20) để không lọc nhầm
        if cb < 0.15 and label_score < DETECT_MIN_LABEL_SCORE:
            logger.debug("filter_background: loại corner box cb=%.2f score=%.2f", cb, label_score)
            continue

        filtered.append(det)
    return filtered


def detect_objects(image_path, min_score=None, max_detections=100,
                   min_label_score=None, min_box_area=None, max_box_area=None):
    min_score       = DETECT_MIN_SCORE       if min_score       is None else min_score
    min_label_score = DETECT_MIN_LABEL_SCORE if min_label_score is None else min_label_score
    min_box_area    = DETECT_MIN_BOX_AREA    if min_box_area    is None else min_box_area
    max_box_area    = DETECT_MAX_BOX_AREA    if max_box_area    is None else max_box_area

    try:
        detections = []
        if USE_SSD_DETECTION:
            detections = detect_objects_ssd(
                image_path, min_score=min_score, max_detections=max_detections,
                min_label_score=min_label_score, min_box_area=min_box_area,
                max_box_area=max_box_area,
            ) or []

        # Bổ sung saliency khi SSD < 3 vật (để bắt vật bị miss)
        if len(detections) < 3 and USE_SALIENCY_DETECTION:
            saliency = detect_objects_saliency(image_path)
            if saliency:
                # Merge và NMS để không duplicate
                combined = detections + saliency
                detections = nms_detections(combined)

        if len(detections) < 1 and GRID_FALLBACK_ENABLED:
            detections = detect_objects_grid_fallback(image_path)

        # Filter theo shape/area/biên (FIX #9: 1 lần duy nhất)
        detections = filter_detection_boxes(detections)
        # Filter background (FIX #9: không check area lần 2)
        detections = filter_background_detections(detections)
        # NMS cuối
        detections = nms_detections(detections)[:MAX_MULTI_DETECTIONS]
        logger.info("Tổng %d đối tượng sau lọc", len(detections))

        # Fallback: if all detections are filtered, fallback to full-image classification
        if len(detections) == 0:
            logger.info("All detections filtered out, fallback to full-image classification.")
        return detections
    except Exception as e:
        logger.exception("detect_objects error: %s", e)
        if GRID_FALLBACK_ENABLED:
            return filter_detection_boxes(detect_objects_grid_fallback(image_path))
        return []


def consolidate_detections(detections):
    grouped = {}
    for det in detections:
        label = det.get('label')
        if not label or label == 'unknown':
            continue
        if label not in grouped:
            grouped[label] = dict(det); grouped[label]['count'] = 1
        else:
            grouped[label]['count'] += 1
            if det.get('score', 0.0) > grouped[label].get('score', 0.0):
                grouped[label]['score'] = det['score']
                grouped[label]['bbox']  = det.get('bbox', grouped[label].get('bbox'))
    return list(grouped.values())


WASTE_LABEL_NORMALIZE = {
    'cardboard': 'paper', 'white-glass': 'glass',
    'brown-glass': 'glass', 'green-glass': 'glass',
}


def build_activity_image_payload(dest_path=None, filename=None):
    image_filename = filename or (os.path.basename(dest_path) if dest_path else None)
    if not image_filename:
        return {'image_path': None, 'image_url': None, 'image_filename': None}
    image_path = f'uploads/{image_filename}'
    return {
        'image_path': image_path,
        'image_url': url_for('static', filename=image_path),
        'image_filename': image_filename,
    }


def log_multi_classification_activities(user_id, detections, dest_path=None, filename=None):
    grouped = consolidate_detections(detections)
    image_payload = build_activity_image_payload(dest_path, filename)
    for det in grouped:
        label = WASTE_LABEL_NORMALIZE.get(det['label'], det['label'])
        for _ in range(max(int(det.get('count', 1)), 1)):
            db.session.add(UserActivity(user_id=user_id, action=f"Phân loại: {label}", **image_payload))


def render_single_classification_result(result, dest_path, filename, is_admin, confidence_score=None):
    waste_details = waste_info.get(result, {})
    recyclability = get_recyclability(result)
    return render_template(
        'result.html',
        result=result,
        confidence_score=confidence_score,
        description=waste_details.get("description", "Thông tin không khả dụng."),
        environmental_impact=waste_details.get("environmental_impact", "Không có thông tin về tác động môi trường."),
        tips=waste_details.get("tips", "Không có mẹo xử lý."),
        recycle_centers=waste_details.get("recycle_centers", []),
        image_url=url_for('static', filename=f'uploads/{filename}'),
        recyclability=recyclability,
        recyclability_description=recyclability_descriptions.get(recyclability, ""),
        is_admin=is_admin,
    )


def build_multi_object_response(dest_path, filename, detections, is_admin):
    display_dets  = sorted(detections, key=lambda d: d.get('label_score', 0.0), reverse=True)
    annotated_name = annotate_detections(dest_path, display_dets, filename)
    grouped = consolidate_detections(detections)
    results = []
    for det in grouped:
        label         = det['label']
        waste_details = waste_info.get(label, {})
        recyclability = get_recyclability(label)
        results.append({
            'label': label, 'score': det['score'], 'count': det.get('count', 1),
            'description':        waste_details.get('description',         'Thông tin không khả dụng.'),
            'environmental_impact': waste_details.get('environmental_impact', ''),
            'tips':               waste_details.get('tips',               ''),
            'recycle_centers':    waste_details.get('recycle_centers',    []),
            'recyclability':      recyclability,
            'recyclability_description': recyclability_descriptions.get(recyclability, ''),
        })
    return render_template(
        'result.html',
        detections=results,
        image_url=url_for('static', filename=f'uploads/{annotated_name}'),
        is_multi=True,
        is_admin=is_admin,
    )


# ───────────────────────── Routes ─────────────────────────

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email',    '').strip()
        phone    = request.form.get('phone',    '').strip()
        address  = request.form.get('address',  '').strip()
        password = request.form.get('password', '')
        
        if not username:
            flash("Username cannot be empty!", "danger"); return redirect('/register')
        if not password:
            flash("Password cannot be empty!", "danger"); return redirect('/register')
        if User.query.filter_by(username=username).first():
            flash("Username already exists!", "danger"); return redirect('/register')
            
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user); db.session.commit()
        db.session.add(CustomerData(user_id=new_user.id, email=email, phone=phone, address=address))
        # Ghi log đăng ký
        db.session.add(UserActivity(user_id=new_user.id, action='Đăng ký tài khoản'))
        db.session.commit()
        flash("Registration successful!", "success"); return redirect('/login')
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user     = User.query.filter_by(username=username).first()
        
        if not user or not check_password_hash(user.password, password):
            flash("Invalid username or password!", "danger")
            return redirect('/login')
        
        # Ghi log đăng nhập
        db.session.add(UserActivity(user_id=user.id, action='Đăng nhập'))
        db.session.commit()
        session['user_id'] = user.id
        return redirect(admin_home_url() if is_admin_user(user) else '/home')
    return render_template('login.html')


@app.route('/logout')
def logout():
    user = get_current_user()
    if user:
        db.session.add(UserActivity(user_id=user.id, action="Logged out"))
        db.session.commit()
    session.pop('user_id', None)
    return redirect('/login')


def get_model_validation_accuracy():
    metrics_path = os.path.join(BASE_DIR, 'model_metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return float(json.load(f).get('val_accuracy', 96.4))
        except Exception:
            pass
    return float(os.environ.get('MODEL_ACCURACY', '96.4'))


def get_model_display_name():
    base = os.path.basename(app.config.get('MODEL_PATH', '')).lower()
    if 'yolo'      in base: return 'YOLOv8'
    if 'mobilenet' in base: return 'MobileNetV2'
    return 'AI Model'


def expand_activity_to_waste_types(action):
    if 'đối tượng' in action:
        n = parse_multi_object_count(action) or 1
        return ['trash'] * n
    waste_type = get_waste_type_from_action(action)
    if not waste_type or waste_type == 'multi':
        return []
    return [WASTE_LABEL_NORMALIZE.get(waste_type, waste_type)]


def build_classification_waste_list():
    waste_list = []
    for activity in get_classification_actions():
        waste_list.extend(expand_activity_to_waste_types(activity.action))
    return waste_list


def count_classified_objects():
    return len(build_classification_waste_list())


def count_classification_sessions():
    activities = (
        UserActivity.query
        .filter(UserActivity.action.like('Phân loại%'))
        .order_by(UserActivity.timestamp.asc())
        .all()
    )
    sessions = 0; i = 0
    while i < len(activities):
        activity = activities[i]
        if 'đối tượng' in activity.action:
            sessions += 1; i += 1; continue
        user_id = activity.user_id
        ts = activity.timestamp.replace(microsecond=0); j = i + 1
        while j < len(activities):
            nxt = activities[j]
            if nxt.user_id != user_id: break
            if 'đối tượng' in nxt.action: break
            if nxt.timestamp.replace(microsecond=0) != ts: break
            if not get_waste_type_from_action(nxt.action): break
            j += 1
        sessions += 1; i = j
    return sessions


def count_classified_images():
    return count_classified_objects()


def format_stat_number(value):
    return f"{value:.1f}" if isinstance(value, float) else f"{int(value):,}".replace(',', '.')


@app.route('/home')
def home():
    user = get_current_user()
    if not user: return redirect('/login')
    if is_admin_user(user): return redirect(admin_home_url())
    model_accuracy = get_model_validation_accuracy()
    stats = {
        'classified_count': format_stat_number(count_classified_images()),
        'model_accuracy':   format_stat_number(model_accuracy) + '%',
        'waste_types_count': format_stat_number(len(waste_types)),
        'model_name':       get_model_display_name(),
    }
    return render_template('home.html', is_admin=False, user=user, stats=stats)


@app.route('/classify')
def classify_redirect():
    return redirect('/')


@app.errorhandler(413)
def request_entity_too_large(_error):
    user = get_current_user()
    return render_template('upload.html', is_admin=is_admin_user(user) if user else False, user=user), 413


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    user = get_current_user()
    if not user: return redirect('/login')
    is_admin = is_admin_user(user)
    if request.method == 'POST':
        mode = request.form.get('mode', 'single')
        try:
            captured_image = request.form.get('captured_image', '').strip()
            file = request.files.get('file')
            has_capture = bool(captured_image)
            has_file = bool(file and file.filename)

            if has_capture and has_file:
                flash("Vui lòng chỉ chọn một trong hai: chụp ảnh trực tiếp hoặc tải file ảnh lên, không được dùng cả hai cùng lúc.", "warning")
            else:
                dest_path, filename = resolve_uploaded_image_from_request()
                if not dest_path:
                    flash("Chưa chọn tệp hoặc chụp ảnh!", "warning")
                else:
                    response = run_classification_for_image(dest_path, filename, user, mode, is_admin)
                    if response is not None:
                        return response
        except ValueError as e:
            logger.warning("Upload/classify validation: %s", e)
        except Exception as e:
            logger.exception("Classification failed: %s", e)
    return render_template('upload.html', is_admin=is_admin, user=user)


@app.route('/leaderboard')
def leaderboard():
    users        = User.query.order_by(User.points.desc()).all()
    current_user = get_current_user()
    return render_template('leaderboard.html', users=users, is_admin=is_admin_user(current_user))


def calculate_activity_streak(activities):
    activity_days = {a.timestamp.date() for a in activities if a.timestamp}
    if not activity_days: return 0
    check = datetime.utcnow().date()
    if check not in activity_days:
        check -= timedelta(days=1)
        if check not in activity_days: return 0
    streak = 0
    while check in activity_days:
        streak += 1; check -= timedelta(days=1)
    return streak


def get_activity_waste_info(action):
    action_lower = action.lower()
    if 'logged out' in action_lower or 'đăng xuất' in action_lower:
        return {'badge': 'logout', 'label': '—', 'points': '—'}
    if not action.startswith('Phân loại'):
        return {'badge': 'other', 'label': 'Khác', 'points': '—'}
    object_count = parse_multi_object_count(action)
    if object_count is not None:
        return {'badge': 'multi', 'label': f'Đa vật thể ({object_count})',
                'points': f'+{points_for_classification(object_count)}'}
    waste_type = action.replace('Phân loại:', '').replace('Phân loại', '').strip()
    if waste_type not in WASTE_TYPE_STYLES:
        waste_type = 'other'
    return {
        'badge':  waste_type,
        'label':  WASTE_TYPE_VN.get(waste_type, waste_type.replace('-', ' ').title()),
        'points': f'+{points_for_classification(1)}',
    }


@app.route('/history')
def history():
    user = get_current_user()
    if not user:
        flash("Please log in to view activity history!", "warning"); return redirect('/login')
    activities       = UserActivity.query.filter_by(user_id=user.id).order_by(UserActivity.timestamp.desc()).all()
    is_admin         = is_admin_user(user)
    classification_count = sum(
        len(expand_activity_to_waste_types(a.action))
        for a in activities if a.action.startswith('Phân loại')
    )
    # Thống kê loại rác được phân loại nhiều nhất
    from collections import Counter
    waste_counter = Counter()
    for a in activities:
        info = get_activity_waste_info(a.action)
        if info.get('badge') and info['badge'] not in ['other', 'logout', 'multi']:
            waste_counter[info['badge']] += 1
    if waste_counter:
        most_common_waste_type = waste_counter.most_common(1)[0][0]
    else:
        most_common_waste_type = '—'
    history_stats = {
        'total_classifications': classification_count,
        'most_common_waste_type': most_common_waste_type,
        'streak_days':          calculate_activity_streak(activities),
        'total_points':         user.points or 0,
    }
    activity_rows = []
    for activity in activities:
        row = get_activity_waste_info(activity.action)
        row['action']    = activity.action
        row['timestamp'] = activity.timestamp.strftime('%d/%m/%Y %H:%M') if activity.timestamp else '—'
        row['image_path'] = getattr(activity, 'image_path', None)
        row['image_url'] = getattr(activity, 'image_url', None)
        row['image_filename'] = getattr(activity, 'image_filename', None)
        if not row['image_url'] and row['image_path']:
            row['image_url'] = url_for('static', filename=row['image_path'])
        if not row['image_url'] and row['image_filename']:
            row['image_url'] = url_for('static', filename=f"uploads/{row['image_filename']}")
        activity_rows.append(row)
    return render_template('history.html', activities=activity_rows,
                           history_stats=history_stats, user=user, is_admin=is_admin)


NEWS_CATEGORIES = [
    {'id': 'all',      'name': 'Tất cả tin tức',       'icon': 'fa-newspaper'},
    {'id': 'domestic', 'name': 'Tin tức trong nước',   'icon': 'fa-flag'},
    {'id': 'world',    'name': 'Tin tức thế giới',     'icon': 'fa-globe'},
    {'id': 'policy',   'name': 'Chính sách - Pháp luật','icon': 'fa-scale-balanced'},
    {'id': 'tech',     'name': 'Công nghệ xanh',       'icon': 'fa-microchip'},
    {'id': 'tips',     'name': 'Hướng dẫn - Tips',     'icon': 'fa-lightbulb'},
    {'id': 'research', 'name': 'Nghiên cứu - Báo cáo', 'icon': 'fa-chart-line'},
]
NEWS_IMAGE_FALLBACK = '/static/uploads/256de136-d5c8-41f2-8044-2f2b523100d21.png'
NEWS_IMAGE_HERO     = '/static/uploads/ChatGPT Image 01_17_50 21 thg 5, 20266.png'


def normalize_news_item(item, hero=False):
    item['image'] = normalize_news_image(item.get('image'), hero=hero)
    item['source_url'] = item.get('source_url') or '#'
    return item


def normalize_news_image(image_url, hero=False):
    if not image_url or str(image_url).startswith('http'):
        return NEWS_IMAGE_HERO if hero else NEWS_IMAGE_FALLBACK
    return image_url


def normalize_news_data(data):
    main = data.get('featured_main')
    if main: normalize_news_item(main, hero=True)
    for item in data.get('featured_side', []): normalize_news_item(item)
    for item in data.get('articles', []):      normalize_news_item(item)
    return data


def get_article_categories(articles):
    grouped = {}
    for article in articles:
        category_name = article.category or 'Khác'
        category_id = slugify_text(category_name)
        if category_id not in grouped:
            grouped[category_id] = {
                'id': category_id,
                'name': category_name,
                'icon': article_icon_for_category(category_name),
                'count': 0,
            }
        grouped[category_id]['count'] += 1
    return sorted(grouped.values(), key=lambda item: (-item['count'], item['name']))


def filter_news_articles(articles, category_id, tab, query):
    filtered = [article for article in articles if article.status == 'published']
    if category_id and category_id != 'all':
        filtered = [article for article in filtered if slugify_text(article.category) == category_id]
    if query:
        query_lower = query.lower()
        filtered = [
            article for article in filtered
            if query_lower in (article.title or '').lower()
            or query_lower in (article.summary or '').lower()
            or query_lower in (article.content or '').lower()
            or query_lower in (article.category or '').lower()
        ]
    if tab == 'popular':
        filtered.sort(key=lambda article: (article.views or 0, article.created_at or datetime.min), reverse=True)
    elif tab == 'featured':
        filtered.sort(key=lambda article: ((article.is_featured or False), article.created_at or datetime.min), reverse=True)
    else:
        filtered.sort(key=lambda article: article.created_at or datetime.min, reverse=True)
    return filtered


def build_news_trending(articles):
    trending = []
    for category in get_article_categories(articles):
        trending.append(category['name'])
        if len(trending) == 5:
            break
    return trending


def build_news_page_context(category_id='all', tab='newest', query=''):
    published_articles = Article.query.filter_by(status='published').all()
    filtered_articles = filter_news_articles(published_articles, category_id, tab, query)
    featured_candidates = [article for article in filtered_articles if article.is_featured]
    featured_main = featured_candidates[0] if featured_candidates else (filtered_articles[0] if filtered_articles else None)
    featured_side = []
    featured_ids = set()
    if featured_main:
        featured_ids.add(featured_main.id)
    for article in filtered_articles:
        if article.id in featured_ids:
            continue
        featured_side.append(article)
        featured_ids.add(article.id)
        if len(featured_side) == 4:
            break
    articles = [article for article in filtered_articles if article.id not in featured_ids]
    return {
        'featured_main': article_to_dict(featured_main) if featured_main else None,
        'featured_side': [article_to_dict(article) for article in featured_side],
        'articles': [article_to_dict(article) for article in articles],
        'categories': get_article_categories(published_articles),
        'trending': build_news_trending(published_articles),
    }


@app.route('/news')
def news_page():
    user = get_current_user()
    if not user:
        flash('Vui lòng đăng nhập!', 'warning'); return redirect('/login')
    category_id = request.args.get('category', 'all')
    tab = request.args.get('tab', 'newest')
    query = request.args.get('q', '').strip()
    context = build_news_page_context(category_id=category_id, tab=tab, query=query)
    tabs = [
        {'id': 'newest', 'name': 'Mới nhất'},
        {'id': 'popular', 'name': 'Xem nhiều'},
        {'id': 'featured', 'name': 'Nổi bật'},
        {'id': 'all', 'name': 'Tất cả'},
    ]
    return render_template(
        'news.html',
        user=user,
        is_admin=is_admin_user(user),
        active_category=category_id,
        active_tab=tab,
        search_query=query,
        tabs=tabs,
        **context,
    )


@app.route('/news/<int:article_id>')
def news_detail_by_id(article_id):
    article = db.session.get(Article, article_id)
    if not article:
        abort(404)
    return redirect(url_for('news_detail', slug=article.slug), code=301)


@app.route('/news/<slug>')
def news_detail(slug):
    user = get_current_user()
    if not user:
        flash('Vui lòng đăng nhập!', 'warning'); return redirect('/login')
    article = Article.query.filter_by(slug=slug).first_or_404()
    if article.status != 'published' and not is_admin_user(user):
        abort(404)
    article.views = (article.views or 0) + 1
    db.session.commit()
    related_articles = (
        Article.query.filter(
            Article.status == 'published',
            Article.category == article.category,
            Article.id != article.id,
        )
        .order_by(Article.is_featured.desc(), Article.created_at.desc())
        .limit(4)
        .all()
    )
    return render_template(
        'news_detail.html',
        user=user,
        is_admin=is_admin_user(user),
        article=article_to_dict(article, include_content=True),
        related_articles=[article_to_dict(item) for item in related_articles],
    )


def get_article_form_choices():
    articles = Article.query.all()
    categories = get_article_categories(articles)
    return categories if categories else [
        {'id': 'moi-truong', 'name': 'Môi trường', 'icon': 'fa-leaf', 'count': 0},
        {'id': 'phan-loai-rac', 'name': 'Phân loại rác', 'icon': 'fa-recycle', 'count': 0},
        {'id': 'ai-va-xu-ly-rac', 'name': 'AI và xử lý rác', 'icon': 'fa-microchip', 'count': 0},
    ]


def save_article_from_form(article=None):
    title = request.form.get('title', '').strip()
    summary = request.form.get('summary', '').strip()
    content = request.form.get('content', '').strip()
    category = request.form.get('category', '').strip()
    author = request.form.get('author', '').strip() or 'Ban biên tập'
    status = request.form.get('status', 'draft').strip() or 'draft'
    is_featured = request.form.get('is_featured') == 'on'
    image_file = request.files.get('image')

    if not title:
        flash('Tiêu đề không được để trống.', 'warning')
        return None
    if not summary:
        flash('Tóm tắt không được để trống.', 'warning')
        return None
    if not content:
        flash('Nội dung không được để trống.', 'warning')
        return None
    if not category:
        flash('Danh mục không được để trống.', 'warning')
        return None

    if article is None:
        image_path = None
    elif isinstance(article, dict):
        image_path = article.get('image_path') or article.get('image')
    else:
        image_path = getattr(article, 'image', None)
    if image_file and image_file.filename:
        image_path = save_article_image_from_upload(image_file)
    elif not image_path:
        image_path = 'uploads/256de136-d5c8-41f2-8044-2f2b523100d21.png'

    if article is None:
        article = Article(
            title=title,
            slug=unique_article_slug(title),
            summary=summary,
            content=content,
            image=image_path,
            category=category,
            category_id=slugify_text(category),
            author=author,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            views=0,
            is_featured=is_featured,
            status=status,
        )
        db.session.add(article)
    else:
        article.title = title
        article.slug = unique_article_slug(title, article.id)
        article.summary = summary
        article.content = content
        article.image = image_path
        article.category = category
        article.category_id = slugify_text(category)
        article.author = author
        article.updated_at = datetime.utcnow()
        article.is_featured = is_featured
        article.status = status
    db.session.commit()
    return article


def delete_article_image_file(image_path):
    if not image_path or str(image_path).startswith('http'):
        return
    relative_path = str(image_path).lstrip('/')
    if relative_path.startswith('static/'):
        relative_path = relative_path[len('static/'):]
    file_path = os.path.join(BASE_DIR, 'static', relative_path)
    if os.path.exists(file_path):
        os.remove(file_path)


@app.route('/admin/articles')
def admin_articles():
    user, denied = require_admin()
    if denied: return denied
    articles = Article.query.order_by(Article.id.asc()).all()
    rows = [article_to_dict(article, include_content=False) for article in articles]
    return render_template('admin_articles.html', user=user, is_admin=True, active_page='articles', articles=rows)


@app.route('/admin/articles/new', methods=['GET', 'POST'])
def admin_article_new():
    user, denied = require_admin()
    if denied: return denied
    if request.method == 'POST':
        article = save_article_from_form()
        if article:
            flash('Đã tạo bài viết mới.', 'success')
            return redirect(url_for('admin_articles'))
    return render_template('admin_article_form.html', user=user, is_admin=True, active_page='articles', article=None, categories=get_article_form_choices())


@app.route('/admin/articles/<int:article_id>/edit', methods=['GET', 'POST'])
def admin_article_edit(article_id):
    user, denied = require_admin()
    if denied: return denied
    article = db.session.get(Article, article_id)
    if not article:
        abort(404)
    if request.method == 'POST':
        updated = save_article_from_form(article)
        if updated:
            flash('Đã cập nhật bài viết.', 'success')
            return redirect(url_for('admin_articles'))
    return render_template('admin_article_form.html', user=user, is_admin=True, active_page='articles', article=article_to_dict(article, include_content=True), categories=get_article_form_choices())


@app.route('/admin/articles/<int:article_id>/delete', methods=['POST'])
def admin_article_delete(article_id):
    user, denied = require_admin()
    if denied: return denied
    article = db.session.get(Article, article_id)
    if not article:
        abort(404)
    delete_article_image_file(article.image)
    db.session.delete(article)
    db.session.commit()
    flash('Đã xóa bài viết.', 'success')
    return redirect(url_for('admin_articles'))


@app.route('/update_customer/<int:id>', methods=['POST'])
def update_customer(id):
    user = get_current_user()
    if not user or not is_admin_user(user): return "Bạn không có quyền!", 403
    customer = CustomerData.query.get_or_404(id)
    customer.email = request.form.get('email')
    customer.phone = request.form.get('phone')
    customer.address = request.form.get('address')
    db.session.commit()
    flash("Cập nhật thành công!", "success")
    return redirect('/admin/users')


WASTE_CATEGORY_GROUPS = {
    'organic':    {'label': 'Rác hữu cơ',  'color': '#22c55e', 'types': {'biological'}},
    'recyclable': {'label': 'Rác tái chế', 'color': '#3b82f6', 'types': {'paper','plastic','metal','glass','clothes'}},
    'inorganic':  {'label': 'Rác vô cơ',   'color': '#f59e0b', 'types': {'trash','shoes'}},
    'hazardous':  {'label': 'Rác nguy hại','color': '#ef4444', 'types': {'battery'}},
}

WASTE_TYPE_VN = {
    'paper':'Giấy','plastic':'Nhựa','metal':'Kim loại','glass':'Thủy tinh',
    'biological':'Rác hữu cơ','trash':'Rác khác','shoes':'Giày dép',
    'clothes':'Quần áo','battery':'Pin','multi':'Đa vật thể',
}

WASTE_TYPE_STYLES = {
    'plastic':    {'bg':'#dbeafe','color':'#2563eb','bar':'#06b6d4'},
    'paper':      {'bg':'#fef3c7','color':'#d97706','bar':'#f59e0b'},
    'metal':      {'bg':'#ffedd5','color':'#c2410c','bar':'#ea580c'},
    'glass':      {'bg':'#ede9fe','color':'#7c3aed','bar':'#8b5cf6'},
    'biological': {'bg':'#dcfce7','color':'#16a34a','bar':'#22c55e'},
    'trash':      {'bg':'#f1f5f9','color':'#475569','bar':'#94a3b8'},
    'shoes':      {'bg':'#fce7f3','color':'#db2777','bar':'#f97316'},
    'clothes':    {'bg':'#fdf2f8','color':'#be185d','bar':'#ec4899'},
    'battery':    {'bg':'#fee2e2','color':'#dc2626','bar':'#ef4444'},
    'multi':      {'bg':'#e0e7ff','color':'#4f46e5','bar':'#6366f1'},
    'other':      {'bg':'#f1f5f9','color':'#64748b','bar':'#94a3b8'},
    'logout':     {'bg':'#f1f5f9','color':'#64748b','bar':'#94a3b8'},
}


def get_waste_style(waste_type):
    return WASTE_TYPE_STYLES.get(waste_type, WASTE_TYPE_STYLES['other'])


def get_waste_type_from_action(action):
    if action.startswith('Phân loại: '):
        return action.replace('Phân loại: ', '').strip()
    return None


def get_classification_actions():
    return UserActivity.query.filter(UserActivity.action.like('Phân loại%')).all()


def get_waste_stats():
    waste_list   = build_classification_waste_list()
    waste_counts = Counter(waste_list)
    total        = sum(waste_counts.values()) or 1
    waste_stats  = [
        {
            'waste_type':    wt,
            'waste_type_vn': WASTE_TYPE_VN.get(wt, wt.replace('-',' ').title()),
            'count':   count,
            'percent': round(count / total * 100, 1),
            'bar_color': get_waste_style(wt)['bar'],
        }
        for wt, count in waste_counts.items() if wt != 'multi'
    ]
    waste_stats.sort(key=lambda x: x['count'], reverse=True)
    return waste_stats, total


def count_supported_waste_types():
    return len([k for k in WASTE_TYPE_VN if k != 'multi'])


def get_waste_category_stats():
    waste_stats, total = get_waste_stats()
    grouped = {key: 0 for key in WASTE_CATEGORY_GROUPS}
    for item in waste_stats:
        wt = item['waste_type']
        for key, meta in WASTE_CATEGORY_GROUPS.items():
            if wt in meta['types']:
                grouped[key] += item['count']; break
    category_stats = []
    for key, meta in WASTE_CATEGORY_GROUPS.items():
        count = grouped[key]
        category_stats.append({
            'key': key, 'label': meta['label'], 'color': meta['color'],
            'count': count, 'percent': round(count / total * 100, 1) if total else 0,
        })
    return category_stats, total


def count_classified_objects_on_date(day):
    activities = UserActivity.query.filter(
        UserActivity.action.like('Phân loại%'),
        func.date(UserActivity.timestamp) == day,
    ).all()
    return sum(len(expand_activity_to_waste_types(a.action)) for a in activities)


def format_time_ago(dt):
    """Định dạng thời gian tương đối tiếng Việt."""
    if not dt:
        return '—'
    now = datetime.utcnow()
    delta = now - dt
    total_seconds = int(delta.total_seconds())
    if total_seconds < 60:
        return 'Vừa xong'
    if total_seconds < 3600:
        return f'{total_seconds // 60} phút trước'
    if total_seconds < 86400:
        return f'{total_seconds // 3600} giờ trước'
    if total_seconds < 172800:  # < 2 ngày
        local_dt = dt + timedelta(hours=7)  # UTC+7
        return f'Hôm qua lúc {local_dt.strftime("%H:%M")}'
    return f'{delta.days} ngày trước'


def get_admin_dashboard_context(user):
    waste_stats, total_objects = get_waste_stats()
    model_accuracy = get_model_validation_accuracy()
    active_users   = User.query.filter(User.username != 'admin').count()
    category_stats, category_total = get_waste_category_stats()
    chart_labels, chart_values = [], []
    for i in range(6, -1, -1):
        day = datetime.utcnow().date() - timedelta(days=i)
        chart_labels.append(day.strftime('%d/%m'))
        chart_values.append(count_classified_objects_on_date(day))

    # Bảng ánh xạ loại hoạt động → icon, tone, message
    ACTIVITY_MAP = {
        'Đăng nhập':        ('fa-right-to-bracket', 'blue',   'đã đăng nhập'),
        'Đăng ký tài khoản':('fa-user-plus',        'green',  'vừa đăng ký tài khoản'),
        'Logged out':       ('fa-right-from-bracket','gray',   'đã đăng xuất'),
    }

    recent_rows = []
    for activity in (UserActivity.query.order_by(UserActivity.timestamp.desc()).limit(30).all()):
        actor    = db.session.get(User, activity.user_id)
        uname    = actor.username if actor else 'Hệ thống'
        time_ago = format_time_ago(activity.timestamp)

        action = activity.action
        if action in ACTIVITY_MAP:
            icon, tone, verb = ACTIVITY_MAP[action]
            message = f'Người dùng <strong>{uname}</strong> {verb}'
        elif action.startswith('Phân loại'):
            icon, tone = 'fa-image', 'purple'
            message = f'Người dùng <strong>{uname}</strong> đã phân loại ảnh'
        else:
            icon, tone = 'fa-clock', 'neutral'
            message = action

        recent_rows.append({'icon': icon, 'tone': tone, 'message': message, 'time_ago': time_ago})
    return {
        'user': user, 'is_admin': True, 'active_page': 'dashboard',
        'kpis': [
            {'label':'Tổng lượt phân loại','value':format_stat_number(total_objects),'trend':'+12.5%','icon':'fa-layer-group','tone':'blue'},
            {'label':'Ảnh đã xử lý','value':format_stat_number(total_objects),'trend':'+15.3%','icon':'fa-image','tone':'green'},
            {'label':'Tỷ lệ nhận diện đúng','value':f'{format_stat_number(model_accuracy)}%','trend':'+2.1%','icon':'fa-bullseye','tone':'orange'},
            {'label':'Người dùng hoạt động','value':format_stat_number(active_users),'trend':'+8.7%','icon':'fa-users','tone':'purple'},
        ],
        'chart_labels': chart_labels, 'chart_values': chart_values,
        'category_stats': category_stats, 'category_total': format_stat_number(category_total),
        'waste_stats': waste_stats[:4], 'recent_rows': recent_rows,
    }


def require_admin():
    user = get_current_user()
    if not user: return None, redirect('/login')
    if not is_admin_user(user): return None, ('Bạn không có quyền!', 403)
    return user, None


def get_customer_data_for_admin():
    users = User.query.order_by(User.id).all()
    for u in users:
        if CustomerData.query.filter_by(user_id=u.id).first() is None:
            db.session.add(CustomerData(user_id=u.id, email='', phone='', address=''))
    db.session.commit()
    return CustomerData.query.join(User).order_by(User.id).all()


@app.route('/admin')
def admin_root():
    user, denied = require_admin()
    if denied: return denied
    return redirect(admin_home_url())


@app.route('/admin/dashboard')
def admin_dashboard():
    user, denied = require_admin()
    if denied: return denied
    return render_template('admin_dashboard.html', **get_admin_dashboard_context(user))


@app.route('/admin/users')
def admin_users():
    user, denied = require_admin()
    if denied: return denied
    return render_template('admin_users.html', user=user, is_admin=True, active_page='users',
                           users=User.query.order_by(User.id).all(),
                           customer_data=get_customer_data_for_admin())


@app.route('/admin/reports')
def admin_reports():
    user, denied = require_admin()
    if denied: return denied
    waste_stats, total = get_waste_stats()
    return render_template('admin_reports.html', user=user, is_admin=True, active_page='reports',
                           waste_stats=waste_stats, total=total,
                           supported_types=count_supported_waste_types(),
                           model_accuracy=get_model_validation_accuracy(),
                           model_name=get_model_display_name())


if __name__ == "__main__":
    try:
        load_detection_model()
    except Exception:
        logger.warning("Detection model will load on first multi-object request")
    if model is None:
        loaded = load_model_safe(MODEL_PATH)
        if loaded is None:
            logger.warning("Model not loaded. Classification will not work until model is placed at MODEL_PATH.")
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
