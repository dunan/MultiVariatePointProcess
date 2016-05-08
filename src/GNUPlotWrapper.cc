/**
 * \file GNUPlotWrapper.cc
 * \brief The class implementation of Plot which is GNUPlot wrapper class.
 */
#include <string>
#include <sstream>
#include "../include/GNUPlotWrapper.h"
#include "../include/Utility.h"
#include "../3rd-party/gnuplot/gnuplot_i.hpp"

void Plot::PlotLinePoint(const std::vector<double>& gp_x, const std::vector<double>& gp_y, const std::vector<double>& gp_point_x, const std::vector<double>& gp_point_y, const std::string& line_title, const std::string& point_title)
{
	Gnuplot::set_terminal_std(driver_);

	Gnuplot g1("lines");

	g1.set_xlabel(x_label_).set_ylabel(y_label_);
	g1.set_style("lines lw 2").plot_xy(gp_x,gp_y,line_title);
	g1.set_style("points ps 1 pt 7").plot_xy(gp_point_x,gp_point_y, point_title);

	wait_for_key();
}

void Plot::PlotLinePoint(const std::vector<std::vector<double> >& gp_x, const std::vector<std::vector<double> >& gp_y, const std::vector<std::vector<double> >& gp_point_x, const std::vector<std::vector<double> >& gp_point_y, const std::vector<std::string>& line_title, const std::vector<std::string>& point_title, const std::vector<std::string>& colors)
{
	Gnuplot::set_terminal_std(driver_);

	Gnuplot g1("lines");

	g1.set_xlabel(x_label_).set_ylabel(y_label_);
	// g1.unset_frame();
	// g1.unset_xtics();
	// g1.unset_ytics();
	std::stringstream css;

	for(unsigned c = 0; c < gp_x.size(); ++ c)
	{
		css.str("");
		css << "lines lw 2 lc rgb" << colors[c];
		g1.set_style(css.str()).plot_xy(gp_x[c],gp_y[c],line_title[c]);
		css.str("");
		css << "points ps 1 linecolor rgb " << colors[c]<< " pt 7";
		g1.set_style(css.str()).plot_xy(gp_point_x[c],gp_point_y[c], point_title[c]);
	}

	wait_for_key();
}

void Plot::PlotScatterLine(const std::vector<double>& gp_x, const std::vector<double>& gp_y, const std::string& line_title)
{
	Gnuplot::set_terminal_std(driver_);

	Gnuplot g1("lines");

	g1.set_xlabel(x_label_).set_ylabel(y_label_);
	g1.set_style("lines lw 2").plot_xy(gp_x,gp_y,line_title);

	wait_for_key();
}