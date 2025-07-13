"""
----------------------------------------
YOLOv8とカメラを使い、危険エリア内の人物を検知・警告し、画像保存・AWS S3アップロード・Slack通知を自動化するリアルタイム危険監視システム。
----------------------------------------
1. カメラ映像をリアルタイムでYOLOv8により人物検知し、指定の「危険エリア」内に人物が入ると「危険」と判定します。
2. 危険検知時は警告音を鳴らし、画面に警告テキストを表示し、直後の画像を保存します。
3. 保存した画像はAWS S3に自動アップロードされ、画像URL付きでSlackにも自動通知します。
4. すべての危険検知履歴はCSVファイルに時刻と画像ファイル名で記録されます。
"""

from ultralytics import YOLO      # YOLOv8用ライブラリのインポート
import cv2                        # OpenCV（画像処理・カメラ操作）をインポート
import os                         # OS操作用モジュール
from datetime import datetime     # 日時操作モジュール
import csv                        # CSV操作モジュール
import time                       # 時間管理用モジュール
import pygame                     # 警告音再生用モジュール
import boto3                      # AWS S3連携用モジュール
import requests                   # HTTPリクエスト用モジュール
from dotenv import load_dotenv    # .envから環境変数読込

load_dotenv(dotenv_path="/****/****/****/.env")  # .envファイル明示読込

# --- セキュアな設定読込 ---
load_dotenv()                                             # .envを再度ロード（パス省略でもOK）
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")        # AWSアクセスキー取得
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")# AWSシークレットキー取得
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")                # S3バケット名取得
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")        # Slack通知用Webhook取得

# --- S3クライアント初期化 ---
s3 = boto3.client(                                        # S3クライアント生成
    "s3",
    region_name="ap-northeast-3",                         # リージョン指定（大阪リージョン例）
    aws_access_key_id=AWS_ACCESS_KEY_ID,                  # アクセスキー指定
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,          # シークレットキー指定
)

# --- Slack通知関数（デバッグ強化） ---
def upload_and_notify(local_path):                        # ファイルをS3アップロードしSlack通知
    if local_path is None:                                # ファイルパス未指定チェック
        raise ValueError("[ERROR] local_path is None.")
    if not os.path.isfile(local_path):                    # ファイル実在チェック
        raise FileNotFoundError(f"[ERROR] File does not exist: {local_path}")
    if AWS_S3_BUCKET is None:                             # S3バケット未設定チェック
        raise ValueError("[ERROR] AWS_S3_BUCKET is None. .env から読み込まれていない可能性があります。")
    if SLACK_WEBHOOK_URL is None:                         # SlackWebhook未設定チェック
        raise ValueError("[ERROR] SLACK_WEBHOOK_URL is None. .env の設定を確認してください。")

    filename = os.path.basename(local_path)               # ファイル名抽出
    s3_key = f"danger_shots/{filename}"                   # S3上のキー名生成

    print(f"[INFO] Uploading {filename} to S3 bucket {AWS_S3_BUCKET}...")  # アップロード開始ログ
    s3.upload_file(local_path, AWS_S3_BUCKET, s3_key)     # S3にアップロード

    presigned_url = s3.generate_presigned_url(            # 有効期限付きURL生成
        "get_object",
        Params={"Bucket": AWS_S3_BUCKET, "Key": s3_key},
        ExpiresIn=3600,
    )

    message = {"text": f"[警告] 人物検知: {presigned_url}"}  # Slack通知用テキスト
    response = requests.post(SLACK_WEBHOOK_URL, json=message) # Slackに通知
    print(f"[INFO] Slack通知ステータス: {response.status_code}") # 通知結果出力

# --- 初期設定 ---
pygame.mixer.init()                                       # pygameのオーディオ初期化
pygame.mixer.music.load("alert.mp3")                      # 警告音ファイル読込

log_file = "danger_log.csv"                               # ログCSVファイル名
save_dir = "danger_shots"                                 # 画像保存ディレクトリ名
os.makedirs(save_dir, exist_ok=True)                      # 保存先ディレクトリ作成（既存OK）

if not os.path.exists(log_file):                          # ログCSVなければ新規作成
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "filename"])        # ヘッダー行のみ書き込み

# --- パラメータ ---
shoot_delay = 0.71                                        # 検知後の撮影ディレイ（秒）
warning_interval = 1.43                                   # 警告インターバル（秒）
last_warning_time = 0                                     # 最終警告時刻（初期値）
CONFIDENCE_THRESHOLD = 0.5                                # 検出信頼度しきい値
DEBUG = False                                             # デバッグ出力フラグ

# --- YOLOv8モデルとカメラ起動 ---
model = YOLO('yolov8n.pt')                                # YOLOv8モデルロード
cap = cv2.VideoCapture(0)                                 # 内蔵カメラ起動

while True:                                               # メインループ
    ret, frame = cap.read()                               # フレーム取得
    if not ret:                                           # 取得失敗時は終了
        break

    results = model(frame)[0]                             # YOLOで推論実行

    danger = False                                        # 危険エリア判定フラグ初期化
    found_person = False                                  # 人物検出フラグ初期化

    h, w = frame.shape[:2]                                # 画像高さ・幅取得
    dz_x, dz_y, dz_w, dz_h = w // 4, h // 3, w // 2, h // 3   # 危険エリア（中央長方形）定義
    cv2.rectangle(frame, (dz_x, dz_y), (dz_x + dz_w, dz_y + dz_h), (0, 0, 255), 2) # 危険エリア枠表示

    for box in results.boxes:                             # 各検出ボックス処理
        cls = int(box.cls[0])                             # クラスID
        conf = float(box.conf[0])                         # 信頼度
        x1, y1, x2, y2 = map(int, box.xyxy[0])            # ボックス座標
        label = results.names[cls]                        # クラス名取得
        if label == "person" and conf > CONFIDENCE_THRESHOLD: # 人物＋信頼度しきい値超え
            found_person = True                           # 人物検出フラグON
            if DEBUG:
                print(f"[DEBUG] class={label}, conf={conf:.2f}, box=({x1},{y1},{x2},{y2})")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 人物検出枠（緑）
            if (x1 < dz_x + dz_w and x2 > dz_x and y1 < dz_y + dz_h and y2 > dz_y): # 危険エリア交差判定
                danger = True                             # 危険フラグON

    current_time = time.time()                            # 現在時刻取得
    if danger and (current_time - last_warning_time >= warning_interval): # 危険かつインターバル経過
        last_warning_time = current_time                  # 最終警告時刻を更新

        pygame.mixer.music.play(loops=0)                  # 警告音再生

        temp = frame.copy()                               # フレームをコピー
        cv2.putText(temp, "!!! DANGER ZONE !!!", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)   # 危険表示テキスト
        cv2.imshow("YOLOv8 Human Detection", temp)        # 危険画面一時表示
        cv2.waitKey(1)

        time.sleep(shoot_delay)                           # 撮影前ディレイ
        ret, shot = cap.read()                            # 再度フレーム取得
        if ret:                                           # 取得成功時のみ
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] # タイムスタンプ生成
            filename = f"{save_dir}/danger_{timestamp}.jpg"             # 保存ファイル名
            resized = cv2.resize(shot, (640, 480))        # 画像リサイズ
            cv2.imwrite(filename, resized, [cv2.IMWRITE_JPEG_QUALITY, 60]) # 画像保存（圧縮）
            with open(log_file, mode='a', newline='') as f:              # ログ追記
                writer = csv.writer(f)
                writer.writerow([timestamp, filename])
            upload_and_notify(filename)                    # S3アップ＆Slack通知

    if DEBUG and found_person:                             # デバッグ時のみ出力
        print(f"[DEBUG] frame: detected person(s)")

    cv2.imshow("YOLOv8 Human Detection", frame)           # メイン画像表示
    if cv2.waitKey(1) & 0xFF == ord('q'):                 # qキーで終了
        break

cap.release()                                             # カメラ解放
cv2.destroyAllWindows()                                   # 全ウィンドウ破棄