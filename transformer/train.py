import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import dataprocessing as dataprocessing
from captioning_solver_transformer import *
from transformer import *
import util as util
import matplotlib.pyplot as plt
import cv2

train_image_path = "../data/smalltrain/*.png"
arr_images, size = dataprocessing.get_pad_images(train_image_path)
train_images = dataprocessing.get_images_tensor(arr_images, size)

train_string_path = "../data/smalltrain_strings/*.tex"
textstrings = dataprocessing.get_pad_strings(train_string_path)
train_strings = dataprocessing.get_text_array(textstrings)
for i in range(len(textstrings)):
    textstrings[i] = ''.join([char for char in textstrings[i] if char != '<NULL>'])

dev_image_path = "../data/dev_images/*.png"
dev_arr_images, dev_size = dataprocessing.get_pad_images(train_image_path)
dev_images = dataprocessing.get_images_tensor(dev_arr_images, dev_size)

dev_string_path = "../data/dev_strings/*.tex"
devcodes = dataprocessing.get_pad_strings(dev_string_path)
dev_strings = dataprocessing.get_text_array(devcodes)
for i in range(len(devcodes)):
    devcodes[i] = ''.join([char for char in devcodes[i] if char != '<NULL>'])

char_to_idx = dataprocessing.create_char_to_idx()
idx_to_char = dataprocessing.create_idx_to_char()
data = util.create_data_dict(train_images, train_strings, textstrings, char_to_idx, idx_to_char, dev_images, dev_strings, devcodes)
device, gpu_ids = util.get_available_devices()

save_dir = "./save/"

transformer = CaptioningTransformer(
          word_to_idx=data['word_to_idx'],
          input_dim=data['train_features'].shape[1],
          wordvec_dim=64,
          num_heads=2,
          num_layers=2,
          max_length=420,
        )

transformer_solver = CaptioningSolverTransformer(transformer, data, idx_to_word=data['idx_to_word'],
           num_epochs=100,
           batch_size=2,
           learning_rate=0.0007,
           verbose=True, print_every=10,
           device=device,
           gpu_ids=gpu_ids,
           save_dir=save_dir,
           eval_steps=700
         )

transformer_solver.train()

print("Dev set BLEU score is ", transformer_solver.evaluate())


# Plot the training losses.
plt.plot(transformer_solver.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.show()

def show_samples():
    sample_captions = transformer.sample(train_images[:3], max_length=420)
    sample_captions = util.decode_codes(sample_captions, data['idx_to_word'])
    i = 5
    for sample_caption in sample_captions:
        
        img = cv2.imread('../data/smalltrain/diagram%d.png' % i, 0)
        i += 1
        
        plt.imshow(img)            
  
        plt.axis('off')
        plt.show()
        print(sample_caption)

# show_samples()