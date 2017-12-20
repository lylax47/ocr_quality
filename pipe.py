import ocr


if __name__ == "__main__":
	ocr.preprocess('00771349.tif', building="word", remove_grey=False)
	metrics = ocr.conf_scores('test.html')
	print(metrics)