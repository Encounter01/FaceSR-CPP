from PIL import Image; img=Image.open('data/train/hr/celeba_00000.png'); lr=img.resize((64,64),Image.BICUBIC); lr.save('data/train/lr/celeba_00000.png'); print('Done')
