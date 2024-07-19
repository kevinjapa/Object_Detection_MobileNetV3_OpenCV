#include <iostream>
#include <fstream>
#include "ObjectDetector.hpp"

int main()
{
	cv::dnn::Net neuralNetwork;
	std::vector<std::string> classes;
	std::string inputDirectory = "../TestFiles";
	std::string outputDirectory = "../Results";

	try{
		neuralNetwork = cv::dnn::readNetFromTensorflow("../SSDMobileNetV3/frozenInterfaceGraph.pb", "../SSDMobileNetV3/frozenInterfaceGraph.pbtxt");
		neuralNetwork.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
		neuralNetwork.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	}
	catch (cv::Exception& error)
	{
		std::cout << error.msg;
		return 1;
	}

	std::ifstream classesFile("../SSDMobileNetV3/classes.txt");
	if (!classesFile.is_open())
		return 1;

	std::string line;
	while (std::getline(classesFile, line))
		classes.push_back(line);


	int processResult = 0;
	ObjectDetector objectDetector = ObjectDetector(neuralNetwork, classes);
	objectDetector.setIODirectory(inputDirectory, outputDirectory);

	//processResult = objectDetector.detectObjects("Catedra-UNESCO-UPS-Imagen-Aula.jpg", ObjectDetector::SourceFileType::Image);
	processResult = objectDetector.detectObjects("Office.mp4", ObjectDetector::SourceFileType::Video);

	return processResult;
}
