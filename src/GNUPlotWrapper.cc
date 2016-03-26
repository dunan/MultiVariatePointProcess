#include <string>
#include "../include/GNUPlotWrapper.h"
#include "../include/Utility.h"
#include "../include/gnuplot/gnuplot_i.hpp"

void Plot::PlotLinePoint(const std::vector<double>& gp_x, const std::vector<double>& gp_y, const std::vector<double>& gp_point_x, const std::vector<double>& gp_point_y, const std::string& point_title)
{
	// Gnuplot::set_terminal_std("wxt size 1280, 800");

	Gnuplot::set_terminal_std(driver_);

	Gnuplot g1("lines");

	g1.set_xlabel(x_label_).set_ylabel(y_label_);
	g1.set_style("lines lw 2").plot_xy(gp_x,gp_y,title_);
	g1.set_style("points ps 2 pt 7").plot_xy(gp_point_x,gp_point_y, point_title);

	wait_for_key();
}

void Plot::PlotScatterLine(const std::vector<double>& gp_x, const std::vector<double>& gp_y)
{
	Gnuplot::set_terminal_std(driver_);

	Gnuplot g1("lines");

	g1.set_xlabel(x_label_).set_ylabel(y_label_);
	g1.set_style("lines lw 2").plot_xy(gp_x,gp_y,title_);

	wait_for_key();
}