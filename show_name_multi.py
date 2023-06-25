import sys
import face_recognition
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import glob

# 設定ファイルの読み込み
import config
threshold = config.thereshold

# 顔情報の初期化
face_locations = []
face_encodings = []

# 登録がぞうの読み込み
image_paths = glob.glob('image/*')
image_paths.sort()
known_face_encodings = [] # 登録済みの画像の配列を用意
known_face_names = [] # 登録済みの画像の名前
checked_face = [] # 一度検知した顔の格納

delimiter = "/"

for image_path in image_paths:
  
  # 拡張子を除いたファイル名を取得
  im_name = image_path.split(delimiter)[-1].split('.')[0]
  # 写真を読み込み
  image = face_recognition.load_image_file(image_path)
  # 写真をエンコード
  face_encoding = face_recognition.face_encodings(image)[0]
  # 読み込んだ画像を配列に追加
  known_face_encodings.append(face_encoding)
  # 読み込んだ画像の名前を配列に追加
  known_face_names.append(im_name)

video_capture = cv2.VideoCapture(0)

def main():
  while True:
    # 処理フラグの初期化
    process_this_frame = True
    # ビデオの単一フレームを取得
    _, frame = video_capture.read()

    if process_this_frame:

      # 顔の位置情報を検索
      face_locations = face_recognition.face_locations(frame)

      # 顔画像の符号化
      face_encodings = face_recognition.face_encodings(frame, face_locations)

      # 名前配列の初期か
      face_names = []

      for face_encoding in face_encodings:
        #　顔画像が登録画像と一致しているか判定
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, threshold) # 判定した結果がthresholdより小さければTrue
        # とりあえず名前を設定
        name = "Unknown"
        # 顔画像と最も近い登録画像を候補とする
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        # 最も似ている画像のマッチがTrueだったら処理に進む
        if matches[best_match_index]:
          name = known_face_names[best_match_index]

        face_names.append(name)
        
      
    # 処理フラグの切り替え
    process_this_frame =  not process_this_frame




    # 位置情報の表示
    for (top, right, bottom, left), name in zip(face_locations, face_names):
      # 顔領域に枠を描画
      cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

      # 顔領域の下に枠を表示
      cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
      font = cv2.FONT_HERSHEY_DUPLEX
      cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

      # 日本語表示
      fontpath = 'meiryo.ttc'


    # 結果をビデオに表示
    cv2.imshow('video', frame)

    # １ミリ秒で処理を実行する
    if cv2.waitKey(1) == 27:
      break

main()

video_capture.release()
cv2.destroyAllWindows()