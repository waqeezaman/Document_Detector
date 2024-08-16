from document_scanner import extract_document
from matplotlib import pyplot as plt
# import boxdetect



from boxdetect.pipelines import get_checkboxes

from boxdetect import config

file_name = 'My_Doc_Examples/form1.png'

cfg = config.PipelinesConfig()

# important to adjust these values to match the size of boxes on your image
cfg.width_range = (30,55)
cfg.height_range = (25,40)

# the more scaling factors the more accurate the results but also it takes more time to processing
# too small scaling factor may cause false positives
# too big scaling factor will take a lot of processing time
cfg.scaling_factors = [0.7]

# w/h ratio range for boxes/rectangles filtering
cfg.wh_ratio_range = (0.5, 1.7)

# group_size_range starting from 2 will skip all the groups
# with a single box detected inside (like checkboxes)
cfg.group_size_range = (2, 100)

# num of iterations when running dilation tranformation (to engance the image)
cfg.dilation_iterations = 0


# limit down the grouping algorithm to just singular boxes (e.g. checkboxes)
cfg.group_size_range = (1, 1)






checkboxes = get_checkboxes(
    file_name, cfg=cfg, px_threshold=0.1, plot=False, verbose=True)


print("Output object type: ", type(checkboxes))
for checkbox in checkboxes:
    print("Checkbox bounding rectangle (x,y,width,height): ", checkbox[0])
    print("Result of `contains_pixels` for the checkbox: ", checkbox[1])
    print("Display the cropout of checkbox:")
    plt.figure(figsize=(1,1))
    plt.imshow(checkbox[2])
    plt.show()































if __name__ == "__main__":

    pass