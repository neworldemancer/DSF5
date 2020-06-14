fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

idxs = np.arange(len(test_labels))
grouped_idx = [idxs[test_labels==i] for i in range(test_labels.max()+1)]
ims_idx = [idx[0] for idx in grouped_idx]
cm = plt.get_cmap()

fig, ax = plt.subplots(2, 6, figsize=(2*6, 2*2))
ax_idxs = [0,1,2,3,4, 6,7,8,9,10, ]
axs = [ax_xy for ax_y in ax for ax_xy in ax_y]
for i, (ax_idx, im_idx) in enumerate(zip(ax_idxs, ims_idx)):
  axi=axs[ax_idx]
  im = test_images[im_idx]
  im_class = test_labels[im_idx]
  axi.imshow(im, cmap='gray')
  axi.text(0, 27, f'{class_names[im_class]}', color='w', size=16)

for axi in axs:
  for axy in [axi.get_xaxis(), axi.get_yaxis()]:
   axy.set_visible(False)
  axi.axis('off')

axi = axs[5]
plt.sca(axi)
for i in range(10):
  plt.scatter([0], [15], c=[cm.colors[256*i//10]], s=200)
plt.scatter([0], [15], c='w', s=180)

fig.legend(class_names, fontsize=16)
plt.tight_layout(0,1,1)




# ====


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
class_names = [str(i) for i in range(10)]
idxs = np.arange(len(test_labels))
grouped_idx = [idxs[test_labels==i] for i in range(test_labels.max()+1)]
ims_idx = [idx[10] for idx in grouped_idx]
cm = plt.get_cmap()

fig, ax = plt.subplots(2, 6, figsize=(2*6, 2*2))
ax_idxs = [0,1,2,3,4, 6,7,8,9,10, ]
axs = [ax_xy for ax_y in ax for ax_xy in ax_y]
for i, (ax_idx, im_idx) in enumerate(zip(ax_idxs, ims_idx)):
  axi=axs[ax_idx]
  im = test_images[im_idx]
  im_class = test_labels[im_idx]
  axi.imshow(im, cmap='gray')
  axi.text(0, 27, f'{class_names[im_class]}', color='w', size=16)

for axi in axs:
  for axy in [axi.get_xaxis(), axi.get_yaxis()]:
   axy.set_visible(False)
  axi.axis('off')

axi = axs[5]
plt.sca(axi)
for i in range(10):
  plt.scatter([0], [15], c=[cm.colors[256*i//10]], s=200)
plt.scatter([0], [15], c='w', s=280)

fig.legend(class_names, fontsize=16)
plt.tight_layout(0,1,1)