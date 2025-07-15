import cv2
import torch

# Charger le modèle YOLOv5 entraîné
model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")
model.conf = 0.4  # seuil de confiance minimal (ajuste selon le besoin)

# Accès à la webcam (0 pour la webcam intégrée)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Utiliser YOLOv5 sur l'image (conversion BGR -> RGB car OpenCV lit en BGR)
    results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Dessiner les résultats sur l'image originale
    annotated_frame = results.render()[0]  # Renvoie une liste d'images

    # Afficher le résultat dans une fenêtre
    cv2.imshow("Fire Detection - YOLOv5", annotated_frame)

    # Quitter avec la touche Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libérer les resources
cap.release()
cv2.destroyAllWindows()
