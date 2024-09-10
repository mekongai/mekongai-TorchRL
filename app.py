from flask import Flask, request, jsonify, render_template
from environment import TicTacToeEnv
from model import initialize_model, train_model

app = Flask(__name__)
env = TicTacToeEnv()

# Khởi tạo mô hình học tăng cường
model, optimizer = initialize_model()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/move', methods=['GET'])
def move():
    row = int(request.args.get('row'))
    col = int(request.args.get('col'))
    _, reward, done, _ = env.step(row * 3 + col)

    if done:
        # Huấn luyện lại mô hình sau mỗi ván chơi
        train_model(model, optimizer, env.memory, reward)

        # Xác định người chiến thắng
        winner = "Người chơi" if reward == 1 else "Máy" if reward == -1 else "Không ai"
        return jsonify({"board": env.board.tolist(), "winner": winner})
    else:
        return jsonify({"board": env.board.tolist(), "winner": None})


if __name__ == '__main__':
    app.run(debug=True)
