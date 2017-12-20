# import the necessary packages
import pytesseract
import pyocr
import cv2
import os
import re
from lxml import etree as et
from PIL import Image
from io import StringIO
import numpy as np


def replace_namespace(xmlstring):
	'''
	replaces all namespaces for easier xml info extraction.
	'''
	newstring = re.sub('\\sxmlns="[^"]+"', '', xmlstring)
	return newstring


def conf_scores(tess_out):
	'''
	outputs average confidence score, variance of conf scores, and buckets containing counts of confidence scores with ranges: x<50, 50<=x<60, 60<=x<70, 70<=x<80, 80<=x

	tess_out - the hocr output created in preprocessing function.
	'''
	out_boxes = []
	buc_less_50 = 0
	buc50 = 0
	buc60 = 0
	buc70 = 0
	buc_great_80 = 0

	with open('test.html') as test:
		test = test.read()
	test = replace_namespace(test)
	test_xml = et.parse(StringIO(test))
	boxes = test_xml.xpath('.//span/@title') #extracts title attribute from all word spans.

	for box in boxes:
		box = box.split('x_wconf ')[1].split('"')[0]
		num = int(box)
		out_boxes.append(num)

		if num < 50:		#calcs distribution of conf scores adn places in buckets.
			buc_less_50 += 1
		elif 50 <= num < 60:
			buc50 += 1
		elif 60 <= num < 70:
			buc60 += 1
		elif 70 <= num < 80:
			buc70 += 1
		elif 80 <= num:
			buc_great_80 += 1


	avg = np.mean(out_boxes) #mean value
	vari = np.std(out_boxes) #standard of deviation

	return (avg, vari, buc_less_50, buc50, buc60, buc70, buc_great_80)


# def bounding_boxes(image): # will add in later if needed. Currently not sure how to implement.




def preprocess(image, lng="eng", building="word", remove_grey=True):
	'''
	Preprocesses image by converting to greyscale, opening, applying gaussianblur, and finally otsu threshhold.

	image - image file in png format

	lng - language, default english

	building - type of desired output, accepts: word, line, char, and text. default is word.

	remove_grey - remove or do not remove greyscale images created in preprocessing. Default is True.
	'''
	im_name = image.split('/')[-1]

	tools = pyocr.get_available_tools()
	tool = tools[0]
	kernel_erode = np.zeros((1,1),np.uint8)
	kernel_dialate = np.zeros((1,1),np.uint8)

	if building == "word":			#decide on build/output type
		build = pyocr.builders.WordBoxBuilder()
	elif building == "line":
		build = pyocr.builders.LineBoxBuilder()
	elif building == "char":
		build = pyocr.tesseract.CharBoxBuilder()
	elif building == "text":
		build = pyocr.builders.TextBuilder()


	image = cv2.imread(image)		#actual preprocessing begins
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.erode(image,kernel_erode,iterations = 2) 
	image = cv2.dilate(image,kernel_dialate,iterations = 2) #opening (erode | dialate)
	image = cv2.GaussianBlur(image,(5,5),0)
	image = cv2.threshold(image,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	# ret, labels = cv2.connectedComponents(image)
	# print(ret)
	# print(labels)


	# crop = image[1372:1503, 1412:1795]
	# cv2.imwrite('crop.jpg', crop)

	filename = "{}.png".format(os.getpid()) #write grayscale
	cv2.imwrite(filename, image)

	boxes = tool.image_to_string(
		Image.open(filename),
		lang=lng,
		builder=build)

	if remove_grey==True:
		os.remove(filename)

	with open("test.html", "w", encoding="utf-8") as file: # prints hocr file.
		build.write_file(file, boxes)