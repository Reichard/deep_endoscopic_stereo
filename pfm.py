from PIL import Image
import re
import numpy as np
import sys

def load_pfm(filename):
	file = open(filename, 'rb')

	color = None
	width = None
	height = None
	scale = None
	endian = None

	header = file.readline().rstrip()
	if header == b'PF':
		color = True
	elif header == b'Pf':
		color = False
	else:
		raise Exception('Not a PFM file.')

	dim_match = re.match(b'^(\d+)\s(\d+)\s$', file.readline())
	if dim_match:
		width, height = map(int, dim_match.groups())
	else:
		raise Exception('Malformed PFM header.')

	scale = float(file.readline().rstrip())
	if scale < 0: # little-endian
		endian = '<'
		scale = -scale
	else:
		endian = '>' # big-endian

	data = np.fromfile(file, endian + 'f')
	shape = (height, width, 3) if color else (height, width)

	data = data.reshape(shape)
	data = np.flipud(data)
	return Image.fromarray(data)

'''
  file = open(filename,'rb')
  color = None
  width = None
  height = None
  scale = None
  endian = None

  header = file.readline().rstrip()
  if header == 'PF':
	color = True    
  elif header == 'Pf':
	color = False
  else:
	raise Exception('Not a PFM file.')

  dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
  if dim_match:
	width, height = map(int, dim_match.groups())
  else:
	raise Exception('Malformed PFM header.')

  scale = float(file.readline().rstrip())
  if scale < 0: # little-endian
	endian = '<'
	scale = -scale
  else:
	endian = '>' # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)
  pixels = np.reshape(data, shape), scale
  return Image.fromarray(pixels)
 '''

