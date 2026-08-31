import os
import io
import base64
import uuid
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from flask import Flask, render_template, request, jsonify, send_file
from utils.coye_filter import apply_coye_filter
import concurrent.futures

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the custom augmentation layer required for loading the models
class SynchronizedAugmentation(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.augmenter = tf.keras.Sequential([
            layers.RandomZoom(height_factor=(-0.2, 0.2)),
            layers.RandomBrightness(factor=0.2),
            layers.RandomContrast(factor=0.2)
        ])

    def call(self, plain_img, coye_img, training=False):
        if not training:
            return plain_img, coye_img
        combined = tf.concat([plain_img, coye_img], axis=-1)
        augmented = self.augmenter(combined, training=True)
        aug_plain = augmented[:, :, :, :3]
        aug_coye = augmented[:, :, :, 3:]
        return aug_plain, aug_coye

    def get_config(self):
        config = super().get_config()
        return config

# Model paths
base_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    'convnext_binary': os.path.join(base_dir, 'modelkeras', 'ConvNextTiny_Binary.keras'),
    'convnext_multi': os.path.join(base_dir, 'modelkeras', 'ConvNeXtTiny_OriginalFix lr4.keras'),
    'effnet_binary': os.path.join(base_dir, 'modelkeras', 'EfficientNetV2S_Binary.keras'),
    'effnet_multi': os.path.join(base_dir, 'modelkeras', 'EfficientNetV2S_OriginalFix lr4.keras')
}

custom_objects = {"SynchronizedAugmentation": SynchronizedAugmentation}

# Cache for Lazy loading
loaded_models = {
    'convnext_binary': None,
    'convnext_multi': None,
    'effnet_binary': None,
    'effnet_multi': None
}

def get_model(choice):
    if choice not in MODEL_PATHS:
        return None, "Invalid model selection"

    if loaded_models[choice] is None:
        print(f"Lazy loading model: {choice}...")
        try:
            loaded_models[choice] = tf.keras.models.load_model(
                MODEL_PATHS[choice],
                custom_objects=custom_objects,
                compile=False
            )
            print(f"{choice} model loaded successfully.")
        except Exception as e:
            print(f"Error loading {choice}: {e}")
            return None, f"Failed to load {choice}: {e}"

    return loaded_models[choice], None


def preprocess_image(img_path):
    """ Reads image from path, runs Coye filter, resizes both to 224x224 and returns batches """
    # Read original image in RGB
    orig_bgr = cv2.imread(img_path)
    if orig_bgr is None:
        raise ValueError("Invalid image file")
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # Get coye filtered image
    coye_rgb = apply_coye_filter(orig_rgb)

    # TF expects float32 or uint8? The notebook used decode_image which returns uint8
    # resize returns float32 but we keep the [0, 255] range as models expect it.
    plain_img = tf.convert_to_tensor(orig_rgb)
    plain_img = tf.image.resize(plain_img, [224, 224])
    
    coye_img = tf.convert_to_tensor(coye_rgb)
    coye_img = tf.image.resize(coye_img, [224, 224])

    # Add batch dimension
    plain_batch = tf.expand_dims(plain_img, 0)
    coye_batch = tf.expand_dims(coye_img, 0)

    # Also save the coye image to static/uploads to display in UI
    filename = str(uuid.uuid4()) + "_coye.jpg"
    coye_save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(coye_save_path, cv2.cvtColor(coye_rgb, cv2.COLOR_RGB2BGR))
    
    return plain_batch, coye_batch, filename

def compute_gradcam(plain_batch, coye_batch, model, model_choice, orig_bgr, save_path):
    import cv2
    import numpy as np
    import tensorflow as tf
    
    if 'convnext' in model_choice.lower():
        try:
            model.get_layer("stream1_convnext")
            stream1_name = "stream1_convnext"
            stream2_name = "stream2_convnext"
            concat_layer_name = "concatenate_1"
        except ValueError:
            stream1_name = "sequential_2"
            stream2_name = "sequential_1"
            concat_layer_name = "concatenate"
        last_conv = "convnext_tiny_stage_3_block_2_identity"
    else:
        try:
            model.get_layer("stream1_effnet")
            stream1_name = "stream1_effnet"
            stream2_name = "stream2_effnet"
            concat_layer_name = "concatenate"
        except ValueError:
            stream1_name = "sequential_2"
            stream2_name = "sequential_1"
            concat_layer_name = "concatenate"
        # Match ConvNeXt by using the final residual block output instead of the 1x1 top_conv
        last_conv = "block6o_add"

    stream1 = model.get_layer(stream1_name)
    base1 = stream1.layers[0]
    target_layer = base1.get_layer(last_conv)
    grad_model_base = tf.keras.Model(inputs=base1.inputs, outputs=[target_layer.output, base1.output])

    with tf.GradientTape() as tape:
        conv_outputs, base1_out = grad_model_base(plain_batch)
        stream1_out = base1_out
        
        stream2 = model.get_layer(stream2_name)
        stream2_out = stream2(coye_batch)
        
        try:
            coye_dense = model.get_layer("coye_reduced_features")
            stream2_out = coye_dense(stream2_out)
        except ValueError:
            pass
            
        concat_layer = model.get_layer(concat_layer_name)
        x = concat_layer([stream1_out, stream2_out])
        
        concat_idx = model.layers.index(concat_layer)
        for layer in model.layers[concat_idx+1:]:
            x = layer(x)
            
        preds = x
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0)
    max_heat = tf.math.reduce_max(heatmap)
    if max_heat != 0:
        heatmap /= max_heat
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (orig_bgr.shape[1], orig_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = jet * 0.4 + orig_bgr * 0.6
    
    cv2.imwrite(save_path, superimposed_img)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'files[]' not in request.files:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        files = [request.files['file']]
    else:
        files = request.files.getlist('files[]')
        
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    model_choice = request.form.get('model', 'convnext_binary')

    # Select model (Lazy loading)
    model, err = get_model(model_choice)
    if model is None:
        return jsonify({'error': err or f'{model_choice} model could not be loaded.'}), 500

    results_json = []
    
    try:
        valid_files = [f for f in files if f.filename != '']
        if not valid_files:
            return jsonify({'error': 'No selected file'}), 400

        file_infos = []
        for file in valid_files:
            # Save uploaded file
            file_id = str(uuid.uuid4())
            orig_filename = file_id + "_orig.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], orig_filename)
            file.save(filepath)
            
            file_infos.append({
                'filename': file.filename,
                'filepath': filepath,
                'file_id': file_id,
                'orig_filename': orig_filename
            })

        def process_single_image(info):
            plain_batch, coye_batch, coye_filename = preprocess_image(info['filepath'])
            return plain_batch, coye_batch, coye_filename, info

        plain_batches = []
        coye_batches = []
        processed_infos = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_single_image, file_infos))
            
        for plain_batch, coye_batch, coye_filename, info in results:
            plain_batches.append(plain_batch)
            coye_batches.append(coye_batch)
            processed_infos.append(info)

        # Batch inference
        if plain_batches:
            all_plain = tf.concat(plain_batches, axis=0)
            all_coye = tf.concat(coye_batches, axis=0)
            
            # Predict all at once
            all_preds = model.predict({'plain_input': all_plain, 'coye_input': all_coye}, batch_size=8)
            
            multi_class_labels = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
            
            for i, info in enumerate(processed_infos):
                preds = all_preds[i]
                is_multi = len(preds) == 5
                
                if is_multi:
                    top_idx = np.argmax(preds)
                    confidence = float(preds[top_idx]) * 100
                    result_class = multi_class_labels[top_idx]
                    prob_dict = {multi_class_labels[j]: f"{float(preds[j])*100:.2f}%" for j in range(5)}
                else:
                    prob_no_dr = float(preds[0])
                    prob_dr = float(preds[1])
                    result_class = "Diabetic Retinopathy (RD)" if prob_dr > prob_no_dr else "No RD"
                    confidence = max(prob_dr, prob_no_dr) * 100
                    prob_dict = {"No RD": f"{prob_no_dr*100:.2f}%", "DR": f"{prob_dr*100:.2f}%"}

                # Grad-CAM
                gradcam_filename = info['file_id'] + "_gradcam.jpg"
                gradcam_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
                orig_bgr = cv2.imread(info['filepath'])
                try:
                    compute_gradcam(plain_batches[i], coye_batches[i], model, model_choice, orig_bgr, gradcam_filepath)
                    gradcam_url = f"/static/uploads/{gradcam_filename}"
                except Exception as e:
                    print(f"Grad-CAM error: {e}")
                    gradcam_url = f"/static/uploads/{info['orig_filename']}" # fallback

                results_json.append({
                    'filename': info['filename'],
                    'prediction': result_class,
                    'confidence': f"{confidence:.2f}%",
                    'is_multi': is_multi,
                    'probabilities': prob_dict,
                    'orig_image_url': f"/static/uploads/{info['orig_filename']}",
                    'gradcam_image_url': gradcam_url
                })

        return jsonify({'results': results_json})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
