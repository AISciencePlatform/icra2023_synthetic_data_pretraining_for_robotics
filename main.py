"""
Copyright (C) 2023 Juan Jose Quiroz Omana
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not,
see <https://www.gnu.org/licenses/>.
"""


import os
from predict import get_predictions


base_path = os.getcwd()

input_path = base_path+"/data/validation_images/imgs"
output_path_unetR = base_path+"/output/UNET_R/"
output_path_unetS = base_path+"/output/UNET_S/"

model_path_unetR = base_path+"/data/models/UNET_R/unet_real.pth"
model_path_unetS = base_path+"/data/models/UNET_S/unet_sim.pth"

get_predictions(input_path, output_path_unetR, model_path_unetR)
get_predictions(input_path, output_path_unetS, model_path_unetS)

print(base_path)