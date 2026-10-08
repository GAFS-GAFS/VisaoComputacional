import cv2
import numpy as np
import glob
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def build_filters():
    """Cria 8 filtros: 7 direcionais (Gabor) e 1 circular para obter 24 dimensões em 3 escalas."""
    filters = []
    ksize = 15 
    
    angles = [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4]
    for theta in angles:
        kern = cv2.getGaborKernel((ksize, ksize), sigma=4.0, theta=theta, lambd=10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
        
    circ_kern = np.array([[ 0, -1,  0],
                          [-1,  4, -1],
                          [ 0, -1,  0]], dtype=np.float32)
    filters.append(circ_kern)
    
    return filters

def center_crop_512(img):
    """Faz um recorte exato de 512x512 no centro da imagem."""
    h, w = img.shape[:2]
    
    if h < 512 or w < 512:
        scale = 512.0 / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
        
    start_y = h//2 - 256
    start_x = w//2 - 256
    
    return img[start_y:start_y+512, start_x:start_x+512]

def extract_features(img_gray, filters):
    img_cropped = center_crop_512(img_gray)
    features = []
    
    img_float = np.float32(img_cropped) / 255.0
    
    for scale in range(3):
        for kern in filters:
            filtered = cv2.filter2D(img_float, cv2.CV_32F, kern)
            mean_val = np.mean(np.abs(filtered))
            features.append(mean_val)
            
        img_float = cv2.pyrDown(img_float)
        
    return np.array(features), img_cropped

def save_clusters_to_folders(images_info, labels, num_clusters, output_dir="resultados_grupos"):
    """Cria pastas para cada grupo e salva as imagens em cinza (512x512) dentro delas."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for c in range(num_clusters):
        cluster_dir = os.path.join(output_dir, f"Grupo_{c+1}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
            
        idx = np.where(labels == c)[0]
        for i in idx:
            name, img = images_info[i]
            save_path = os.path.join(cluster_dir, name)
            cv2.imwrite(save_path, img) # Salva a imagem processada fisicamente no disco
            
    print(f"\nSucesso! As {len(images_info)} imagens foram separadas nas pastas dentro de: '{output_dir}'")

def process_and_cluster(image_folder, num_clusters=4):
    image_paths = glob.glob(os.path.join(image_folder, "*.*"))
    image_paths = sorted(list(set([p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))])))[:32]
    
    if not image_paths:
        print("Nenhuma imagem encontrada na pasta.")
        return

    print(f"Extraindo atributos de {len(image_paths)} imagens...")
    filters = build_filters()
    features_list = []
    images_loaded = []
    
    for path in image_paths:
        img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        vec, cropped_img = extract_features(img_gray, filters)
        features_list.append(vec)
        images_loaded.append((os.path.basename(path), cropped_img))
        
    X = np.array(features_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Agrupando texturas...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Chama a nova função para salvar em pastas em vez de exibir na tela
    save_clusters_to_folders(images_loaded, labels, num_clusters)

if __name__ == "__main__":
    pasta_imagens = "./suas_imagens" 
    process_and_cluster(pasta_imagens, num_clusters=4)