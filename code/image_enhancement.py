import cv2
import glob

# CLAHE
# https://github.com/alec-ng/fully-convolutional-network-semantic-segmentation/blob/master/scripts/clahe.py

def enhance(image_path, clip_limit=3):
    image = cv2.imread(image_path)
    # convert image to LAB color model
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # split the image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    # apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    file = 'images/clahe.jpg'
    cv2.imwrite(file, cl)
    
    # merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge((cl, a_channel, b_channel))

    # convert iamge from LAB color model back to RGB color model
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return final_image


# set-up directories for input and output
path = '../seed images/input/indx/'  
path_output = '../seed images/output/indx/'
files = glob.glob(path + '*.jpg')   
imgcount = 0 # initialization


for file in files:
    print(imgcount)
    print(file)
    
    # enhance image using CLAHE
    res = enhance(file)
    
    # enalrge using BICUBIC image interpolation
    bicubic_img = cv2.resize(res, None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
    
    cv2.imwrite(path_output + str(imgcount) + '.jpg', bicubic_img)
    
    imgcount = imgcount + 1

print('DONE')