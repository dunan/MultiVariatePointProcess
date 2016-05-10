/**
 * \file Utility.cc
 * \brief Contains the implementation of auxiliary I/O functions.
 */
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

void ImportFromExistingCascades(const std::string& filename, const unsigned& number_of_nodes, double& T, std::vector<Sequence>& data)
{
	std::ifstream fin(filename.c_str());
	std::string str;
	unsigned line = 0;
	unsigned seqID = 0;

    double maxT = 0;

	while(std::getline(fin, str))
	{
		if((++ line) > number_of_nodes + 1)
		{

			std::vector<std::string> parts = SeperateLineWordsVector(str, ",");

			Sequence seq;

			for(unsigned i = 0; i < parts.size(); i += 2)
			{
				unsigned eventID = 0;

				Event event;
				event.EventID = (eventID ++);
				event.SequenceID = seqID;
				event.DimentionID = std::atoi(parts[i].c_str());
				event.time = std::atof(parts[i+1].c_str());
				event.marker = -1;

                if(maxT < event.time)
                {
                    maxT = event.time;
                }
				seq.Add(event);
			}

			data.push_back(seq);
			++ seqID;

		}
		
	}
	fin.close();

	for(unsigned c = 0; c < data.size(); ++ c)
    {
        data[c].SetTimeWindow(maxT);
        // std::cout << data[c].GetEvents().size() << " " << data[c].GetTimeWindow() << std::endl;
    }

    T = maxT;
}

void ImportFromExistingSingleSequence(const std::string& filename, Sequence& seq)
{
    std::ifstream fin(filename.c_str());
    std::string str;
    unsigned seqID = 0;
    std::getline(fin, str);
    std::vector<std::string> parts = SeperateLineWordsVector(str, " ");
    unsigned eventID = 0;
    unsigned count = 0;
    for(std::vector<std::string>::const_iterator i_timing = parts.begin(); i_timing != parts.end(); ++ i_timing)
    {
        Event event;
        event.EventID = (eventID ++);
        event.SequenceID = seqID;
        event.DimentionID = 0;
        event.time = atof(i_timing->c_str());
        event.marker = -1;
        seq.Add(event);
        if(++count > 100000)
        {
            break;
        }
    }
    
    fin.close();
}

void ImportFromExistingSequences(const std::string& filename, std::vector<Sequence>& data, double scale)
{
    std::ifstream fin(filename.c_str());
    std::string str;
    unsigned seqID = 0;

    std::cout << filename << std::endl;

    while(std::getline(fin, str))
    {
        std::vector<std::string> parts = SeperateLineWordsVector(str, " ");

        double origin = atof(parts[0].c_str());

        Sequence seq;

        unsigned eventID = 0;

        for(std::vector<std::string>::const_iterator i_timing = parts.begin(); i_timing != parts.end(); ++ i_timing)
        {
            Event event;
            event.EventID = (eventID ++);
            event.SequenceID = seqID;
            event.DimentionID = 0;
            event.time = (atof(i_timing->c_str()) - origin) / scale;
            event.marker = -1;
            seq.Add(event);
        }

        data.push_back(seq);
        ++ seqID;
        
    }

    fin.close();

    std::cout << seqID << std::endl;

}

void ImportFromExistingUserItemSequences(const std::string& filename, const unsigned& num_users, const unsigned& num_items, std::vector<Sequence>& data)
{
    std::ifstream fin(filename.c_str());
    std::string str;
    unsigned seqID = 0;
    while(std::getline(fin, str))
    {
        std::vector<std::string> parts = SeperateLineWordsVector(str, "\t");
        unsigned i = atoi(parts[0].c_str()) - 1;
        unsigned j = atoi(parts[1].c_str()) - 1;
        unsigned dim_id = i + j * num_users;

        std::vector<std::string> timings = SeperateLineWordsVector(parts[2], " ");

        Sequence seq;

        for(std::vector<std::string>::const_iterator i_timing = timings.begin(); i_timing != timings.end(); ++ i_timing)
        {
            unsigned eventID = 0;

            Event event;
            event.EventID = (eventID ++);
            event.SequenceID = seqID;
            event.DimentionID = dim_id;
            event.time = atof(i_timing->c_str());
            event.marker = -1;
            seq.Add(event);
        }

        data.push_back(seq);
        ++ seqID;

        // std::cout << dim_id << " " << seq.GetEvents().size() << " " << seq.GetTimeWindow() << std::endl;

    }
    fin.close();
}

void LoadEigenMatrixFromTxt(const std::string& filename, const unsigned& num_rows, const unsigned& num_cols, Eigen::MatrixXd& mat)
{
    std::ifstream fin(filename.c_str());
    std::string str;

    std::vector<double> elements;

    while(std::getline(fin, str))
    {
        elements.push_back(atof(str.c_str()));
    }

    fin.close();

    mat = Eigen::Map<Eigen::MatrixXd>(elements.data(), num_rows, num_cols);  
    
}

void wait_for_key()
{
    std::cout << std::endl << "Press ENTER to continue..." << std::endl;
    std::cin.clear();
    std::cin.ignore(std::cin.rdbuf()->in_avail());
    std::cin.get();
}

double PowerMethod(const Eigen::MatrixXd& M, unsigned it_max, double tol, Eigen::VectorXd& u, Eigen::VectorXd& v)
{

    Eigen::MatrixXd MM = M.transpose() * M;
    unsigned n = MM.rows();
    if(MM.isZero())
    {
        u = Eigen::MatrixXd::Identity(n,1);
        v = u;
        return 0;
    }

    
    Eigen::VectorXd x = (Eigen::VectorXd::Random(n).array() + 1) * 0.5;
    double s, s_old;
    v = x.array() / x.norm();
    s_old = (M * v).norm();
    for(unsigned i = 0; i < it_max; ++ i)
    {
        x = MM * x;
        v = x.array() / x.norm();
        s = (M * v).norm();
        if (std::fabs(s - s_old) < tol)
        {
            break;
        }
        s_old = s;
    }

    v = x.array() / x.norm();

    Eigen::VectorXd temp = M * v;
    s = temp.norm();
    u = temp.array() / s;

    return s;
}

double SimpsonIntegral38(FunctionHandler& functor, double a, double b, unsigned n)
{
    Eigen::VectorXd t = Eigen::VectorXd::Zero(n + 1);
    Eigen::VectorXd y = Eigen::VectorXd::Zero(n + 1);

    double h = (b - a) / double(n);

    for(unsigned i = 0; i < n + 1; ++ i)
    {
        t(i) = a + i * h;
    }

    functor(t, y);

    double integral = 0;

    for(unsigned i = 1; i < n; ++ i)
    {
        if(i % 3 == 0)
        {
            integral += 2 * y(i);
        }else
        {
            integral += 3 * y(i);
        }
    }

    integral = (3 * h / 8) * (y(0) + y(n) + integral);

    return integral;

}
