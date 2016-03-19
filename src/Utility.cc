#include <fstream>
#include <iostream>
#include <cstdlib>
#include "../include/Utility.h"

std::vector<std::string> SeperateLineWordsVector(const std::string &lineStr, const std::string& splitter)
{
    std::string::size_type pos = 0;
    std::string::size_type prePos = 0;
    std::string filtChars(splitter);
    std::string lastWord = "";
    std::string tempWord;

    int count = 0;
    std::vector<std::string> temp;

    while((pos = lineStr.find_first_of(filtChars,pos)) != std::string::npos)
    {
        if(lineStr.substr(pos, filtChars.size()) == filtChars)
        {
            count++;
            tempWord = "";
            tempWord = lineStr.substr(prePos,pos - prePos);
            if (tempWord != "")
            {
                temp.push_back(tempWord);
            }
            pos += filtChars.size();
            prePos = pos;

        }else
        {
            ++ pos;
        }
    }
    if(prePos < lineStr.length())
    {
        temp.push_back(lineStr.substr(prePos,lineStr.length() - prePos));
    }

    return temp;
}

void ImportFromExistingCascades(const std::string& filename, const unsigned& number_of_nodes, const double& T, std::vector<Sequence>& data)
{
	std::ifstream fin(filename.c_str());
	std::string str;
	unsigned line = 0;
	unsigned seqID = 0;
	while(std::getline(fin, str))
	{
		if((++ line) > number_of_nodes + 1)
		{

			std::vector<std::string> parts = SeperateLineWordsVector(str, ",");

			Sequence seq(T);

			for(unsigned i = 0; i < parts.size(); i += 2)
			{
				unsigned eventID = 0;

				Event event;
				event.EventID = (eventID ++);
				event.SequenceID = seqID;
				event.DimentionID = std::atoi(parts[i].c_str());
				event.time = std::atof(parts[i+1].c_str());
				event.marker = -1;

				seq.Add(event);
			}

			data.push_back(seq);
			++ seqID;

			std::cout << seq.GetEvents().size() << " " << seq.GetTimeWindow() << std::endl;

		}
		
	}
	fin.close();

	std::cout << data.size() << std::endl;
}