#ifndef GNUPLOT_WRAPPER_H
#define GNUPLOT_WRAPPER_H

#include <vector>
#include <string>

class Plot
{

private:

	std::string x_label_, y_label_, driver_;

public:

	Plot(const std::string& driver, const std::string& x_label, const std::string& y_label) : x_label_(x_label), y_label_(y_label), driver_(driver){}

	void PlotLinePoint(const std::vector<double>& gp_x, const std::vector<double>& gp_y, const std::vector<double>& gp_point_x, const std::vector<double>& gp_point_y, const std::string& line_title, const std::string& point_title);

	void PlotScatterLine(const std::vector<double>& gp_x, const std::vector<double>& gp_y, const std::string& line_title);

	void PlotLinePoint(const std::vector<std::vector<double> >& gp_x, const std::vector<std::vector<double> >& gp_y, const std::vector<std::vector<double> >& gp_point_x, const std::vector<std::vector<double> >& gp_point_y, const std::vector<std::string>& line_title, const std::vector<std::string>& point_title, const std::vector<std::string>& colors);

};

#endif