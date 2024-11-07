import librosa
import soundfile as sf
import os
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter

#Define the augmentation process
augment = Compose([
    AddGaussianNoise(min_amplitude=0.1, max_amplitude=0.2, p=0.5),
    PitchShift(min_semitones=-8, max_semitones=8, p=0.5),
    HighPassFilter(min_cutoff_freq=2000, max_cutoff_freq=4000, p=0.5)
])

def augment_audio(input_file, output_file):
    """
    Load an audio file, apply augmentations, and save the augmented audio.
    
    Parameters:
        input_file (str): Path to the input audio file.
        output_file (str): Path to the output augmented audio file.
    """
    #Load the audio file
    signal, sr = librosa.load(input_file)
    
    #Apply the augmentation
    augmented_signal = augment(signal, sr)
    
    #Save the augmented audio to the specified output file
    sf.write(output_file, augmented_signal, sr)

if __name__ == "__main__":
    #Prompt the user for the input file path
    file_path = input("Please enter the path to the audio file: ")
    
    #Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File '{file_path}' not found. Please check the path and try again.")
    else:
        #Define the output file name
        output_path = "augmented_" + os.path.basename(file_path)  # Creates an output file name based on the input
        
        try:
            #Call the augment_audio function
            augment_audio(file_path, output_path)
            print(f"Augmented audio saved as {output_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
