/**
 * \file GNUPlotWrapper.h
 * \brief The class definition of Plot which is GNUPlot wrapper class.
 */
#ifndef GNUPLOT_WRAPPER_H
#define GNUPLOT_WRAPPER_H

#include <vector>
#include <string>

/**
 * \class Plot GNUPlotWrapper.h "include/GNUPlotWrapper.h"
 * \brief Plot is a wrapper of the GNU plot.
 */
class Plot
{

private:

/**
 * \brief Label of the x-axis
 */
	std::string x_label_;
/**
 * \brief Label of the y-axis
 */
	std::string y_label_;
/**
 * \brief The driver used to produce the plot
 */
	std::string driver_;

public:

/**
 * The constructor
 *
 * @param[in] driver the driver used to produce the plot
 * @param[in] label of the x-axis
 * @param[in] label of the y-axis
 */
	Plot(const std::string& driver, const std::string& x_label, const std::string& y_label) : x_label_(x_label), y_label_(y_label), driver_(driver){}

/**
 * \brief Plot line and dots
 * @param gp_x        x coordinates of the line plot
 * @param gp_y        y coordinates of the line plot
 * @param gp_point_x  x coordinates of the dot plot
 * @param gp_point_y  y coordinates of the dot plot
 * @param line_title  legend of the line plot
 * @param point_title legend of the point plot
 */
	void PlotLinePoint(const std::vector<double>& gp_x, const std::vector<double>& gp_y, const std::vector<double>& gp_point_x, const std::vector<double>& gp_point_y, const std::string& line_title, const std::string& point_title);

/**
 * \brief Plot lines
 * @param gp_x       x coordinates of the line plot
 * @param gp_y       y coordinates of the line plot
 * @param line_title legend of the line plot
 */
	void PlotScatterLine(const std::vector<double>& gp_x, const std::vector<double>& gp_y, const std::string& line_title);

/**
 * \brief Plot line and dots
 * @param gp_x        x coordinates of the line plot
 * @param gp_y        y coordinates of the line plot
 * @param gp_point_x  x coordinates of the dot plot
 * @param gp_point_y  y coordinates of the dot plot
 * @param line_title  legend of the line plot
 * @param point_title legend of the point plot
 * @param colors      colors of lines and points
 */
	void PlotLinePoint(const std::vector<std::vector<double> >& gp_x, const std::vector<std::vector<double> >& gp_y, const std::vector<std::vector<double> >& gp_point_x, const std::vector<std::vector<double> >& gp_point_y, const std::vector<std::string>& line_title, const std::vector<std::string>& point_title, const std::vector<std::string>& colors);

};

#endif