import matlab.engine
import os
path = './05_Walking_Towards_radar'
path2 = './pictures'
eng = matlab.engine.start_matlab()
eng.addpath(path)
os.chdir(path2)

for file in os.listdir(f"{path}"):
    if file.endswith('.bin'):
        eng.datToImage_77( file, file, 'test_picture', nargout=0)
eng.quit()