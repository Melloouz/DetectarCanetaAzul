import cv2
import numpy as np

# 1. Inicializar a webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()

print("pressione q para sair")

while True:
    # 1 Captura frame a frame
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame")
        break
    
    # 2. Converta o frame para o espaço de cores HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 3. Define os limites de cor para detectar a caneta (exemplo : azul)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # 4. Cria uma máscara para a cor azul
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Aplicar a máscara para reduzir ruidos
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 5. identificar contornos na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Se houver contornos, desenhe o maior contorno na imagem original
    if len(contours) > 0:
        # Pega o maior contorno
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        # Filtrar por area mínima para evitar ruídos
        if area > 500:
            # Obtém a caixa delimitadora do contorno
            x, y, w, h = cv2.boundingRect(c)

            #desenha um retângulo verde ao redor da caneta
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Adiciona um rótulo "Caneta" ao Retângulo
            cv2.putText(frame, "Caneta", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    # 7. exibe o frame resultante
    cv2.imshow("Detecção de Caneta", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Libera a webcam e fecha as janelas
cap.release()
cv2.destroyAllWindows()