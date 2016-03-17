#ifndef UTILITY_H
#define UTILITY_H

constexpr unsigned int string2int(const char *str, int h = 0)
{
    // Hash function
    return !str[h] ? 5381 : (string2int(str, h+1)*33) ^ str[h];
}


#endif