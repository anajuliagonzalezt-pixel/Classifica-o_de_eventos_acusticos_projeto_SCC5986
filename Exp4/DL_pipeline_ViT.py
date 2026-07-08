import os
import gc
import json
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time

from sklearn.model_selection import StratifiedKFold, ParameterGrid, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    balanced_accuracy_score,
    confusion_matrix,
    top_k_accuracy_score
)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from transformers import TFViTModel

#Para imprimir as GPUs encontradas
print(tf.config.list_physical_devices('GPU'))

# Garante o determinismo para os testes não ficarem mudando de resultado
def definir_sementes(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

definir_sementes(42)

# Diretorios para organizar saídas e caches
DATASET_PATH = "DATASEC/"
CACHE_DIR = "Espectrogramas_Cache/"
OUTPUT_CSV = "resultados_validacao_grid_search_loss.csv"
TEST_FINAL_CSV = "resultado_teste_final_loss_campeao.csv"
PRED_DIR = "Predicoes_Outputs/"
GRAFICOS_DIR = "Graficos_Treinamento/"
MATRIZES_DIR = "Matrizes_Confusao/"

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(GRAFICOS_DIR, exist_ok=True)
os.makedirs(MATRIZES_DIR, exist_ok=True)

# Desativa a interface gráfica do matplotlib pra não estourar memória no servidor
plt.switch_backend('Agg')


#Geração dos espectrogramas
def extract_melspectrogram(file_path, sr=22050, n_mels=224, max_len=224):
    # Carrega o áudio completo usando o sample rate reduzido (22050 Hz), pois acima disso não existe muito som significativo, e o tamanho do sample rate afeta o output do espectrograma
    y, sr = librosa.load(file_path, sr=sr)
    
    # Força o retorno de audios para aproximadamente 5.18s:
    # 114176 dividido por 22050 = aprox 5.18
    # Escolhemos 114176 porque dividido pelo hop_length (512) dá exatamente 223.
    # Com o +1 da fórmula do STFT, o Librosa gera exatamente 224 colunas de tempo.
    # Importante ser 224x224 para bater com o tamanho de entrada das redes
    target_samples = 114176
    novo_hop = 512
    
    # Padding
    if len(y) < target_samples:
        # Se o áudio for menor que o esperado, preenche o final com silêncio (zeros)
        pad_len = target_samples - len(y)
        y = np.pad(y, (0, pad_len), mode='constant')
    else:
        # Se o áudio for maior que o esperado, pegamos o trecho central.
        centro = len(y) // 2
        inicio = centro - (target_samples // 2)
        y = y[inicio:inicio+target_samples]

    # Extração de mel espec
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=novo_hop)
    
    # Converte a escala de potência para Decibéis
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Garantindo que não houve nenhum erro de formação do arquivo no tamanho que esperamos
    if mel_db.shape[1] < max_len:
        pad = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad)), mode='constant', constant_values=np.min(mel_db))
    else:
        mel_db = mel_db[:, :max_len]
        
    return mel_db


# Criação de classe da Focal Loss para otimização da rede, visto que não existe esse otimizador pronto no tensorflow/keras
class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=None, gamma=2.0, name="categorical_focal_loss"):
        super(CategoricalFocalLoss, self).__init__(name=name)
        self.gamma = float(gamma)
        # Se passar uma lista ou array de pesos, vira um tensor constante
        self.alpha = tf.constant(alpha, dtype=tf.float32) if alpha is not None else None

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = tf.math.pow(1.0 - y_pred, self.gamma)
        loss = focal_weight * cross_entropy
        
        if self.alpha is not None:
            loss = self.alpha * loss
            
        return tf.math.reduce_sum(loss, axis=-1)
    
class ClassBalancedLoss(tf.keras.losses.Loss):
    def __init__(self, samples_per_cls, beta=0.999, name="class_balanced_loss"):
        super(ClassBalancedLoss, self).__init__(name=name)
        # Calcula os pesos pelo volume efetivo de amostras (Cui et al., 2019)
        effective_num = [1.0 - beta ** n for n in samples_per_cls]
        weights = [(1.0 - beta) / en for en in effective_num]
        # Normaliza para que a soma dos pesos seja igual ao número de classes
        total = sum(weights)
        weights = [w * len(samples_per_cls) / total for w in weights]
        self.class_weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        # Aplica o peso da classe correspondente
        loss = self.class_weights * cross_entropy
        return tf.math.reduce_sum(loss, axis=-1)

#Calculo da F1-Macro feito de forma "manual" devido a problemas de incompatibilidade de versões de algumas biliotecas 
class F1MacroMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="f1_macro", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred_labels = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=self.num_classes)

        self.tp.assign_add(tf.reduce_sum(y_true * y_pred_labels, axis=0))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred_labels, axis=0))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred_labels), axis=0))

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        f1_per_class = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return tf.reduce_mean(f1_per_class)

    def reset_state(self):
        self.tp.assign(tf.zeros((self.num_classes,)))
        self.fp.assign(tf.zeros((self.num_classes,)))
        self.fn.assign(tf.zeros((self.num_classes,)))


X_raw, y_raw = [], []
#Carrega os dados dos arquivos de cache (caso já tenhamos computado e salvado os especs anteriormente)
if os.path.exists(CACHE_DIR) and len(os.listdir(CACHE_DIR)) > 0:
    print(" Cache de espectrogramas encontrado! Carregando arquivos...")
    X_raw = np.load(os.path.join(CACHE_DIR, "X_data.npy"))
    y_raw = np.load(os.path.join(CACHE_DIR, "y_labels.npy"))
else:
    #Se não, calcula e salva os especs como .npy
    print(" Processando os arquivos originais do DataSEC...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                last_folder = os.path.basename(root)
                try:
                    spec = extract_melspectrogram(file_path)
                    X_raw.append(spec)
                    y_raw.append(last_folder)
                except Exception as e:
                    print("Erro no arquivo:", file_path, e)
    
    X_raw = np.array(X_raw)
    y_raw = np.array(y_raw, dtype=str)
    np.save(os.path.join(CACHE_DIR, "X_data.npy"), X_raw)
    np.save(os.path.join(CACHE_DIR, "y_labels.npy"), y_raw)

#Faz o encoding dos labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
X = np.array(X_raw)[..., np.newaxis]  
num_classes = len(label_encoder.classes_)

#Separa o conjunto de teste e treinamento (de forma estratificada) antes de qualquer normalização nos dados
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)


#Definição dos modelos
#Fizemos 3 modelos de CNN: um simples (2 camadas conv), intermediario (4 camadas conv) e um complexo (6 camadas conv)
#E também 2 modelos de transfer learning com camadas intermediarias das redes: Vision Transformer e ResNet-50
#E um modelo ViT treinado do 0
def create_model(input_shape, num_classes, learning_rate=0.001, conv_filters=16, 
                 dropout_rate=0.0, kernel_size=(3,3), optimizer_type='adam', 
                 model_type='simple_cnn', feature_layer_name=None, dense_units=128,
                 loss_strategy='categorical_crossentropy', alpha_weights=None, samples_per_cls=None):
    
    norm_layer = layers.Normalization(axis=None, name="normalization_layer")
    model_inputs = layers.Input(shape=input_shape, name="input_main")
    x = norm_layer(model_inputs)
    
    # Pequeno data augmentation nos dados de treino por meio de deslocamento horizontal simples (Time-shift)
    # Possivelmente ajuda classes minoritárias com menos de 200 instâncias
    x = layers.RandomTranslation(height_factor=0.0, width_factor=0.025, fill_mode='constant', name="time_shift")(x)
    
    if model_type == 'simple_cnn':
        x = layers.Conv2D(conv_filters, kernel_size, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Conv2D(conv_filters * 2, kernel_size, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(dense_units, activation='relu')(x)
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=model_inputs, outputs=outputs, name="Simple_CNN")

    elif model_type == 'intermediate_cnn':
        x = layers.Conv2D(conv_filters, kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(conv_filters, kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        if dropout_rate > 0.0: x = layers.Dropout(dropout_rate * 0.5)(x)

        x = layers.Conv2D(conv_filters * 2, kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(conv_filters * 2, kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        if dropout_rate > 0.0: x = layers.Dropout(dropout_rate)(x)

        x = layers.GlobalAveragePooling2D()(x) 
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=model_inputs, outputs=outputs, name="Intermediate_CNN")
    elif model_type == 'complex_cnn':
        x = layers.Conv2D(conv_filters, kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(conv_filters, kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        if dropout_rate > 0.0: x = layers.Dropout(dropout_rate * 0.5)(x)

        x = layers.Conv2D(conv_filters * 2, kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(conv_filters * 2, kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        if dropout_rate > 0.0: x = layers.Dropout(dropout_rate)(x)

        x = layers.Conv2D(conv_filters * 4, kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(conv_filters * 4, kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        if dropout_rate > 0.0: x = layers.Dropout(dropout_rate)(x)

        # Cabeça de classificação com dois Dense antes da softmax
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        if dropout_rate > 0.0: x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units // 2, activation='relu')(x)
        if dropout_rate > 0.0: x = layers.Dropout(dropout_rate * 0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=model_inputs, outputs=outputs, name="Complex_CNN")
    elif model_type == 'vit_intermediaria':

        # Triplica os canais para RGB (Exigência padrão de ViTs)
        x_rgb = layers.Concatenate(axis=-1)([x, x, x])

        # Redimensiona para o tamanho esperado pelo ViT do HuggingFace (224x224)
        x_resized = layers.Resizing(224, 224)(x_rgb)

        # HuggingFace espera canais em formato (batch, canais, altura, largura) -> "channels_first"
        # então precisamos transpor antes de passar pro modelo
        x_chw = layers.Permute((3, 1, 2))(x_resized)

        # Instancia o ViT pré-treinado na ImageNet-21k
        base_vit = TFViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            output_hidden_states=True,
            use_safetensors=False
        )
        base_vit.trainable = False  # Congela os pesos para Transfer Learning

        # Passa a imagem pelo Transformer, pedindo os hidden_states de todas as camadas
        vit_outputs = base_vit(pixel_values=x_chw, training=False)

        camada_escolhida = 8
        estado_intermediario = vit_outputs.hidden_states[camada_escolhida]

        # Extração do embedding (token CLS, primeira posição da sequência)
        x_embedding = estado_intermediario[:, 0, :]

        # Finaliza a classificação usando MLP
        x_mlp = layers.Dense(dense_units, activation='relu')(x_embedding)
        if dropout_rate > 0.0:
            x_mlp = layers.Dropout(dropout_rate)(x_mlp)

        outputs = layers.Dense(num_classes, activation='softmax')(x_mlp)
        model = models.Model(inputs=model_inputs, outputs=outputs, name="ViT_Intermed_HF")

    elif model_type == 'resnet_intermediaria':
        x_rgb = layers.Concatenate(axis=-1)([x, x, x])
        
        #Carrega o modelo completo da ResNet50
        full_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
        full_resnet.trainable = False
        
        # Escolhemos a camada de corte (Fim do Bloco 6)
        # Camada antes da final, com representações menos complexas / mais genericas
        camada_corte = 'conv4_block6_out' 
        
        # Criamos um submodelo que vai da entrada até a camada definida
        base_model = models.Model(
            inputs=full_resnet.input, 
            outputs=full_resnet.get_layer(camada_corte).output,
            name="ResNet50_Intermediaria"
        )
        
        # Passa o áudio pelo novo modelo
        features = base_model(x_rgb, training=False)
        x_embeddings = layers.GlobalAveragePooling2D()(features)
        
        # Finaliza a classificação usando MLP
        x_mlp = layers.Dense(dense_units, activation='relu')(x_embeddings)
        if dropout_rate > 0.0: x_mlp = layers.Dropout(dropout_rate)(x_mlp)
        outputs = layers.Dense(num_classes, activation='softmax')(x_mlp)
        model = models.Model(inputs=model_inputs, outputs=outputs, name="ResNet_Intermed_Audio")
    elif model_type == 'vit_completa':
        # Triplica os canais para RGB (Exigência padrão de ViTs)
        x_rgb = layers.Concatenate(axis=-1)([x, x, x])

        # Redimensiona para o tamanho esperado pelo ViT (224x224)
        x_resized = layers.Resizing(224, 224)(x_rgb)

        # HuggingFace espera "channels_first" -> (batch, canais, altura, largura)
        x_chw = layers.Permute((3, 1, 2))(x_resized)

        # Instancia o ViT pré-treinado na ImageNet-21k
        base_vit = TFViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            output_hidden_states=False, 
            use_safetensors=False
        )
        
        # Descongela a rede inteira para treinamento real
        base_vit.trainable = True 

        # Passa os dados pelo Transformer completo
        vit_outputs = base_vit(pixel_values=x_chw, training=True)

        # Pega a saída da ÚLTIMA camada (shape: [batch_size, sequence_length, hidden_size])
        ultimo_estado_oculto = vit_outputs.last_hidden_state

        # Extração do token CLS (primeira posição da sequência) que representa o áudio inteiro
        x_embedding = ultimo_estado_oculto[:, 0, :]

        # Cabeça de classificação (MLP)
        x_mlp = layers.Dense(dense_units, activation='relu')(x_embedding)
        if dropout_rate > 0.0:
            x_mlp = layers.Dropout(dropout_rate)(x_mlp)

        outputs = layers.Dense(num_classes, activation='softmax')(x_mlp)
        model = models.Model(inputs=model_inputs, outputs=outputs, name="ViT_Completa_FineTuning")

    #2 opções de otimizadores, apesar de que depois decidimos por focar apenas no Adam 
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) if optimizer_type == 'adam' else tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    
    # 3 estrategias de calculo de loss, uma tradicional (categorical_crossentropy) e outras customizadas, que possivelmente funcionam melhor para dados desbalanceados
    if loss_strategy == 'focal_loss':
        selected_loss = CategoricalFocalLoss(alpha=alpha_weights, gamma=2.0)
    elif loss_strategy == 'class_balanced_loss':
        selected_loss = ClassBalancedLoss(samples_per_cls=samples_per_cls)
    else:
        selected_loss = 'categorical_crossentropy'

    # Retorna os modelos com os parametros escolhidos e que gera metricas de acuracia e f1-macro
    model.compile(
        optimizer=opt,
        loss=selected_loss,
        metrics=['accuracy', F1MacroMetric(num_classes=num_classes, name='f1_macro')]
    )
    return model, norm_layer


# Gera gráfico do treinamento do modelo
def plot_training_history(history, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], label='Treino')
    axes[0].plot(history.history['val_loss'], label='Validação')
    axes[0].set_title('Histórico de Loss')
    axes[0].legend()
    
    axes[1].plot(history.history['f1_macro'], label='Treino')
    axes[1].plot(history.history['val_f1_macro'], label='Validação')
    axes[1].set_title('Histórico de F1-Macro')
    axes[1].legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


#Grid de hiperparametros
#Inicialmente iriamos testar vários hiperparametros para cada rede, mas seria inviavel em termos de tempo.
param_grid = [
    {
        'model_type': ['vit_completa'], 
        'learning_rate': [2e-5],  # Taxa bem menor e segura para ajuste de Transformers
        'batch_size': [16],       # Reduzido para 16 porque a ViT completa consome MUITA memória de GPU
        'conv_filters': [0], 
        'kernel_size': [None], 
        'dropout_rate': [0.3], 
        'optimizer_type': ['adam'], 
        'feature_layer_name': [None], 
        'dense_units': [128], 
        'loss_strategy': ['class_balanced_loss']
    }
]

grid = ParameterGrid(param_grid)
resultados_lista = []

# Como é apenas uma configuração da ViT Completa, fazemos um hold-out direto 
# Separamos 12.5% do X_dev para validação (o que equivale a 10% do total dos dados)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.125, random_state=42, stratify=y_dev
)

# Faz o treinamento para cada modelo do grid (neste caso, apenas a ViT)
for params in grid:
    print(f"\n TREINANDO MODELO ÚNICO: {params['model_type'].upper()} | Loss: {params['loss_strategy']}")
    start_time = time.time()
    
    # Calcula a distribuição de classes com base no treino atual
    classes_unicas, contagens = np.unique(y_train, return_counts=True)
    samples_per_cls = [contagens[i] for i in range(len(classes_unicas))]
    
    current_class_weight = None # Mantido apenas para compatibilidade de assinatura se necessário
    alpha_focal = None
    samples_per_cls_param = samples_per_cls if params['loss_strategy'] == 'class_balanced_loss' else None
    
    # Define o modelo com os parâmetros do grid
    model, norm_layer = create_model(
        input_shape=X_train.shape[1:], num_classes=num_classes,
        learning_rate=params['learning_rate'], conv_filters=params['conv_filters'],
        dropout_rate=params['dropout_rate'], kernel_size=params['kernel_size'],
        optimizer_type=params['optimizer_type'], model_type=params['model_type'],
        feature_layer_name=params['feature_layer_name'], dense_units=params['dense_units'],
        loss_strategy=params['loss_strategy'], alpha_weights=alpha_focal,
        samples_per_cls=samples_per_cls_param
    )

    norm_layer.adapt(X_train)
    
    # Early stopping monitorando val_loss
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)

    # Treina o modelo em uma única rodada direta
    history = model.fit(X_train, to_categorical(y_train, num_classes=num_classes),
                        validation_data=(X_val, to_categorical(y_val, num_classes=num_classes)),
                        epochs=150, batch_size=params['batch_size'], 
                        callbacks=[early_stop], verbose=1)
    
    # Gera as predições no conjunto de validação para o relatório intermediário
    y_prob_val = model.predict(X_val, verbose=0)
    y_pred_val = np.argmax(y_prob_val, axis=1)
    
    # Calcula as métricas de validação
    val_acc = accuracy_score(y_val, y_pred_val)
    val_bal_acc = balanced_accuracy_score(y_val, y_pred_val)
    val_f1  = f1_score(y_val, y_pred_val, average='macro')
    val_top3 = top_k_accuracy_score(y_val, y_prob_val, k=3, labels=np.arange(num_classes)) if num_classes > 3 else 1.0
    val_top5 = top_k_accuracy_score(y_val, y_prob_val, k=5, labels=np.arange(num_classes)) if num_classes > 5 else 1.0
    
    end_time = time.time()
    duration_seconds = end_time - start_time
    print(f"       Tempo de Treinamento: {int(duration_seconds // 60)}m {int(duration_seconds % 60)}s")
    print(f"    [VALIDAÇÃO]: F1-Macro={val_f1:.4f} | Balanced-Acc={val_bal_acc:.4f} | Acc={val_acc:.4f}")
    
    # Salva o gráfico histórico deste treinamento
    plot_training_history(history, os.path.join(GRAFICOS_DIR, "curva_treino_vit_completa.png"))
    
    # Armazena os resultados de validação no CSV padrão de logs
    resultados_lista.append({
        'model_type': params['model_type'], 'loss_strategy': params['loss_strategy'],
        'mean_val_accuracy': val_acc, 'mean_val_balanced_accuracy': val_bal_acc,
        'mean_val_f1_macro': val_f1, 'mean_val_top_3_accuracy': val_top3, 'mean_val_top_5_accuracy': val_top5
    })
    pd.DataFrame(resultados_lista).to_csv(OUTPUT_CSV, index=False)

    # -----------------------------------------------------------------------------
    #  AVALIAÇÃO DIRETA NO CONJUNTO DE TESTE REAL (SEM RETREINAR)
    # -----------------------------------------------------------------------------
    print("\n PREDICT NOS DADOS DE TESTE REAIS...")
    y_test_prob = model.predict(X_test, verbose=0)
    y_test_pred = np.argmax(y_test_prob, axis=1)

    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    test_acc = accuracy_score(y_test, y_test_pred)
    test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    test_top3 = top_k_accuracy_score(y_test, y_test_prob, k=3, labels=np.arange(num_classes)) if num_classes > 3 else 1.0
    test_top5 = top_k_accuracy_score(y_test, y_test_prob, k=5, labels=np.arange(num_classes)) if num_classes > 5 else 1.0

    print(f"\n [RESULTADOS NOS DADOS DE TESTE REAIS]:")
    print(f"    F1-Macro Final:        {test_f1:.4f}")
    print(f"    Acurácia Balanceada:   {test_bal_acc:.4f}")
    print(f"    Acurácia Global:       {test_acc:.4f}")

    # Geração automática da matriz de confusão para os slides
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Matriz de Confusão - ViT Completa Otimizada')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.savefig(os.path.join(MATRIZES_DIR, "matriz_confusao_final.png"), bbox_inches='tight')
    plt.close()

    # Salva os resultados detalhados das predições de teste no CSV final
    df_test_final = pd.DataFrame({
        'true_numeric': y_test, 'true_name': label_encoder.inverse_transform(y_test),
        'pred_numeric': y_test_pred, 'pred_name': label_encoder.inverse_transform(y_test_pred),
        'test_accuracy': test_acc, 'test_balanced_accuracy': test_bal_acc, 'test_f1_macro': test_f1,
        'test_top_3_accuracy': test_top3, 'test_top_5_accuracy': test_top5
    })
    df_test_final.to_csv(TEST_FINAL_CSV, index=False)
    
    # Limpeza de memória ao final do loop
    del model; tf.keras.backend.clear_session(); gc.collect()

print("\n Script finalizado com sucesso!")