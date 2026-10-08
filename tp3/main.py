"""
Integrantes:
Gabriel Augusto Fabri Soltovski / GRR20222546
Ricardo Quer / GRR20224827

"""

import cv2
import numpy as np
import argparse
import sys
import os

def order_points(pts):
    """
    Ordena 4 pontos cartesianos na ordem:
    [superior-esquerdo (tl), superior-direito (tr), inferior-direito (br), inferior-esquerdo (bl)].
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Superior-esquerdo terá a menor soma (x + y), inferior-direito terá a maior soma
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Superior-direito terá a menor diferença (y - x), inferior-esquerdo terá a maior diferença
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts):
    """
    Calcula a matriz de transformação perspectiva (homografia) e aplica warpPerspective
    para obter a vista frontal retificada da região selecionada.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calcula a largura da nova imagem
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # Calcula a altura da nova imagem
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Define os pontos de destino no plano frontal ortogonal
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Calcula a matriz de perspectiva 3x3 e realiza a interpolação espacial
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))
    
    return warped

def apply_scanner_effect(warped_image):
    """
    Converte a imagem retificada em escala de cinza e aplica limiarização adaptativa
    para simular o efeito de um documento escaneado (preto e branco de alto contraste).
    """
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11
    )
    return scanned

def manual_select_points(image):
    """
    Abre uma janela para que o usuário selecione manualmente os 4 cantos do documento
    com cliques do mouse caso a detecção automática falhe ou seja solicitado.
    """
    pts = []
    clone = image.copy()
    window_name = "Selecione 4 cantos (Clique em cada canto | 'r' reinicia | 'ESC' cancela)"

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([x, y])
            cv2.circle(clone, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(clone, f"P{len(pts)}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if len(pts) > 1:
                cv2.line(clone, tuple(pts[-2]), tuple(pts[-1]), (0, 255, 255), 2)
            if len(pts) == 4:
                cv2.line(clone, tuple(pts[3]), tuple(pts[0]), (0, 255, 255), 2)
            cv2.imshow(window_name, clone)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)
    cv2.imshow(window_name, clone)

    print("\n[INFO] Modo Manual Ativado:")
    print("  -> Clique com o botão esquerdo nos 4 cantos do documento.")
    print("  -> Pressione 'r' para reiniciar os pontos se errar.")
    print("  -> Pressione qualquer outra tecla quando concluir os 4 pontos.")

    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('r'):
            pts.clear()
            clone = image.copy()
            cv2.imshow(window_name, clone)
            print("[INFO] Pontos resetados. Clique novamente nos 4 cantos.")
        elif key == 27:  # ESC
            print("[AVISO] Seleção cancelada pelo usuário.")
            cv2.destroyWindow(window_name)
            return None
        elif len(pts) == 4:
            cv2.waitKey(500)
            break

    cv2.destroyWindow(window_name)
    return np.array(pts, dtype="float32")

def auto_detect_document_corners(image):
    """
    Tenta detectar automaticamente os 4 vértices do documento através de detecção de bordas
    e aproximação poligonal dos maiores contornos.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # Dilatar levemente para fechar eventuais quebras no contorno
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
            
    return None

def main():
    ap = argparse.ArgumentParser(
        description="Aplicação de Transformação de Perspectiva (Scanner de Documentos) - TP3"
    )
    ap.add_argument("-i", "--image", required=True, help="Caminho para a imagem de entrada")
    ap.add_argument("-o", "--output", default=None, help="Caminho para salvar a imagem retificada resultante")
    ap.add_argument("-m", "--manual", action="store_true", help="Força a seleção manual dos 4 pontos com o mouse")
    ap.add_argument("-s", "--scan-effect", action="store_true", help="Gera também o efeito de scanner P&B com limiarização")
    ap.add_argument("--no-display", action="store_true", help="Não abre janelas na interface (modo headless)")
    args = vars(ap.parse_args())

    # 1. Carregar a imagem
    image_path = args["image"]
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERRO] Não foi possível carregar a imagem '{image_path}'. Verifique o caminho.")
        sys.exit(1)

    orig = image.copy()
    corners = None

    # 2. Obtenção dos 4 vértices
    if not args["manual"]:
        print("[INFO] Tentando detectar as 4 bordas do documento automaticamente...")
        corners = auto_detect_document_corners(image)
        if corners is not None:
            print("[SUCESSO] Contorno do documento de 4 pontos detectado com sucesso!")
        else:
            print("[AVISO] Não foi possível detectar automaticamente as 4 bordas com precisão.")
            if not args["no_display"]:
                print("[INFO] Iniciando modo de seleção manual como alternativa...")
                corners = manual_select_points(image)
            else:
                print("[ERRO] Em modo --no-display, a detecção automática falhou e não há interface para seleção manual.")
                sys.exit(1)
    else:
        if not args["no_display"]:
            corners = manual_select_points(image)
        else:
            print("[ERRO] Não é possível usar o modo --manual em conjunto com --no-display.")
            sys.exit(1)

    if corners is None or len(corners) != 4:
        print("[ERRO] O processo foi encerrado sem os 4 pontos necessários.")
        sys.exit(1)

    # 3. Aplicar a transformação perspectiva
    warped = four_point_transform(orig, corners)

    # 4. Gerar efeito de scanner P&B se solicitado
    scanned = None
    if args["scan_effect"]:
        scanned = apply_scanner_effect(warped)

    # 5. Salvar resultados em disco se especificado
    if args["output"]:
        output_path = args["output"]
        cv2.imwrite(output_path, warped)
        print(f"[SALVO] Imagem retificada salva em: {output_path}")

        if scanned is not None:
            base, ext = os.path.splitext(output_path)
            scanned_path = f"{base}_scanner{ext if ext else '.png'}"
            cv2.imwrite(scanned_path, scanned)
            print(f"[SALVO] Imagem com efeito scanner salva em: {scanned_path}")

    # 6. Exibição na tela
    if not args["no_display"]:
        # Desenha o polígono detectado na cópia da imagem original para visualização didática
        display_orig = orig.copy()
        pts_poly = order_points(corners).astype(int).reshape((-1, 1, 2))
        cv2.polylines(display_orig, [pts_poly], isClosed=True, color=(0, 255, 0), thickness=3)
        for i, pt in enumerate(order_points(corners).astype(int)):
            cv2.circle(display_orig, tuple(pt), 6, (0, 0, 255), -1)

        cv2.imshow("1. Imagem Original (Regiao Detectada)", display_orig)
        cv2.imshow("2. Perspectiva Retificada (Colorida)", warped)
        if scanned is not None:
            cv2.imshow("3. Efeito Scanner (P&B Binarizado)", scanned)

        print("\n[INFO] Pressione qualquer tecla na janela da imagem para encerrar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()