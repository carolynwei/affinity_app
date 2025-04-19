import logging
from flask import Flask, request, jsonify, render_template
from model_core import predict_affinity
import boto3
import os

# 初始化 Flask 应用
app = Flask(__name__)

# 设置日志级别为 DEBUG
app.logger.setLevel(logging.DEBUG)

# 创建日志处理器：输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 添加处理器到 Flask 的日志系统
app.logger.addHandler(console_handler)


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
        app.logger.error(f"预测失败: {str(e)}")
        return jsonify({'error': str(e)}), 500  # 出错时返回错误信息

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Render 会自动设置 PORT 环境变量
    app.run(host='0.0.0.0', port=port)
