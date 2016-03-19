#ifndef UTILITY_H
#define UTILITY_H
#include "Process.h"
#include <string>
#include <vector>



// constexpr unsigned int string2int(const char *str, int h = 0)
// {
//     // Hash function
//     return !str[h] ? 5381 : (string2int(str, h+1)*33) ^ str[h];
// }


void ImportFromExistingCascades(const std::string& filename, const unsigned& number_of_nodes, const double& T, std::vector<Sequence>& data);

std::vector<std::string> SeperateLineWordsVector(const std::string &lineStr, const std::string& splitter);


#endif