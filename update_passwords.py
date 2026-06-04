import os
from werkzeug.security import generate_password_hash
from app import app, db, User

with app.app_context():
    users = User.query.all()
    hashed_pwd = generate_password_hash('1111')
    
    updated_count = 0
    for user in users:
        user.password = hashed_pwd
        updated_count += 1
            
    db.session.commit()
    print(f"Đã cập nhật mật khẩu mặc định (1111) cho {updated_count} tài khoản.")
