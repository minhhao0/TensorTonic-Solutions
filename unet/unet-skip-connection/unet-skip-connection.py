import numpy as np
def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Crop encoder features to match decoder spatial dims, then concatenate along channels.
    """
    # Your implementation here
    encoder_features=np.array(encoder_features)
    decoder_features=np.array(decoder_features)
    b,he,we,c=encoder_features.shape
    b,hd,wd,c=decoder_features.shape
    sigmoid_h=(he-hd)//2
    sigmoid_w=(we-wd)//2
    crop_features=encoder_features[:,sigmoid_h:sigmoid_h+hd,sigmoid_w:sigmoid_w+wd,:]
    print(crop_features.shape)
    return np.concatenate((crop_features,decoder_features),axis=3)