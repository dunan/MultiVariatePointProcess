#ifndef GNUPLOT_WRAPPER_H
#define GNUPLOT_WRAPPER_H

#include <vector>
#include <string>

class Plot
{

private:

	std::string x_label_, y_label_, title_, driver_;

public:

	Plot(const std::string& driver, const std::string& x_label, const std::string& y_label, const std::string& title) : x_label_(x_label), y_label_(y_label), title_(title), driver_(driver){}

	void PlotLinePoint(const std::vector<double>& gp_x, const std::vector<double>& gp_y, const std::vector<double>& gp_point_x, const std::vector<double>& gp_point_y, const std::string& point_title);

	void PlotScatterLine(const std::vector<double>& gp_x, const std::vector<double>& gp_y);

};

#endif