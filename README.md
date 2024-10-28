# AdaFace + Whisper Finetuning:
Relies on https://github.com/mk-minchul/AdaFace and https://github.com/openai/whisper/

# Whisper 
pip install -U openai-whisper

Whisper is downloaded automatically. And after istalling it via pip the script (should) run without any problems (it is VERY demanding hardware wise atleast on my pc)

# Adaface
You will currently have to download the code from the repository via git and then just drag the file in there or adjust the depency. https://github.com/mk-minchul/AdaFace

You will need to download the Adaface model here https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view different version can be found on their github


# Install
First Download the adaface github: 
```
git clone git@github.com:mk-minchul/AdaFace.git
```
of if you prefer https:
```
git clone https://github.com/mk-minchul/AdaFace.git
```
Then navigate into the folder and create a conda evironment:
```
cd AdaFace
conda create --name MAI pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
conda activate MAI
conda install scikit-image matplotlib pandas scikit-learn
cd ..
```
Lastly install the requirements. The requirements.txt is untested but should work.
```
pip install -r requirements.txt
```
