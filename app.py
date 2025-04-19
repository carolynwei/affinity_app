# app.py
from flask import Flask, request, jsonify, render_template
from model_core import predict_affinity
import boto3
import os

def download_model_from_s3(bucket_name, s3_key, local_path):
    try:
        print("🛠️ 开始下载模型...", flush=True)
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        if not os.path.exists(local_path):
            print(f"📦 Downloading {s3_key} from S3...", flush=True)
            s3.download_file(bucket_name, s3_key, local_path)
            print(f"✅ Downloaded to {local_path}", flush=True)
        else:
            print(f"✅ Found cached model at {local_path}", flush=True)
    except Exception as e:
        print("❌ 下载模型失败：", e, flush=True)

# 初始化 Flask 应用
app = Flask(__name__)

# ✅ 日志：确认正在开始模型下载
print("🧪 准备调用 download_model_from_s3", flush=True)

# 设置 bucket 和模型路径
bucket_name = 'my-antibody-app'
s3_key_1 = 'model/model1231_epoch30.pth'
local_path_1 = os.path.join(os.path.dirname(__file__), 'model1231_epoch30.pth')

s3_key_2 = 'model/pretrain_bert.models'
local_path_2 = os.path.join(os.path.dirname(__file__), 'pretrain_bert.models')

# 下载两个模型
download_model_from_s3(bucket_name, s3_key_1, local_path_1)
download_model_from_s3(bucket_name, s3_key_2, local_path_2)

print("✅ download_model_from_s3 已被调用完成", flush=True)

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')

# 预测 API 路由：处理前端 POST 请求
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    seq_light = data['light']
    seq_heavy = data['heavy']
    seq_antigen = data['antigen']

    try:
        # 预测模型函数，返回亲和力得分
        score = predict_affinity(seq_light, seq_heavy, seq_antigen)
        return jsonify({'affinity': score})  # 👈 这个字段要和前端对应
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # 出错时返回错误信息

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Render 会自动设置 PORT 环境变量
    app.run(host='0.0.0.0', port=port)
