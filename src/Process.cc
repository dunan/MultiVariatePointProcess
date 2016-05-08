/**
 * \file Process.cc
 * \brief The class implementation of Process which defines the general interface of a point process.
 */
#include "../include/Process.h"
#include "../include/GNUPlotWrapper.h"

void IProcess::InitializeDimension(const std::vector<Sequence>& data)
{
	unsigned num_sequences_ = data.size();

	all_timestamp_per_dimension_ = std::vector<std::vector<std::vector<double> > >(num_sequences_, std::vector<std::vector<double> > (num_dims_, std::vector<double> ()));

	for(unsigned c = 0; c < num_sequences_; ++ c)
	{
		const std::vector<Event>& seq = data[c].GetEvents();

		for(unsigned i = 0; i < seq.size(); ++ i)
		{
			all_timestamp_per_dimension_[c][seq[i].DimentionID].push_back(seq[i].time);
		}

	}
}

void IProcess::PlotIntensityFunction(const Sequence& data)
{
	double delta = 0.01;

	unsigned max_dim = 4;

	unsigned num_plot_dim = (num_dims_ < max_dim ? num_dims_ : max_dim);

	const double& Tc = data.GetTimeWindow();

	unsigned num_points = unsigned(Tc / delta);

	Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(num_points, 0, Tc);

	Eigen::MatrixXd y = Eigen::MatrixXd::Zero(num_plot_dim, num_points);

	Eigen::VectorXd intensity_dim(num_dims_);

	for(unsigned i = 0; i < num_points; ++ i)
	{
		Intensity(t(i), data, intensity_dim);
		y.col(i) = intensity_dim.segment(0, num_plot_dim);
	}

	std::vector<std::vector<double> > gp_x(num_plot_dim, std::vector<double>(num_points, 0));
	std::vector<std::vector<double> > gp_y(num_plot_dim, std::vector<double>(num_points, 0));
	std::vector<std::string> line_title, point_title;

	for(unsigned c = 0; c < num_plot_dim; ++ c)
	{
		for(unsigned i = 0; i < num_points; ++ i)
		{
			gp_x[c][i] = t(i);
			gp_y[c][i] = y(c,i);
		}
		std::stringstream css;
		css << "Intensity of dimension" << " " << c;
		line_title.push_back(css.str());
		css.str("");
		css << "Events on dimension" << " " << c;
		point_title.push_back(css.str());
	}

	const std::vector<Event>& events = data.GetEvents();

	std::vector<std::vector<double> > gp_x_point(num_plot_dim, std::vector<double>());
	std::vector<std::vector<double> > gp_y_point(num_plot_dim, std::vector<double>());

	for(std::vector<Event>::const_iterator i_event = events.begin(); i_event != events.end(); ++ i_event)
	{
		gp_x_point[i_event->DimentionID].push_back(i_event->time);
		gp_y_point[i_event->DimentionID].push_back(-0.2 * (i_event->DimentionID + 1));
	}

	std::vector<std::string> colors = {"'dark-orange'", "'blue'", "'dark-red'", "'dark-spring-green'"};

	Plot plot("wxt size 640, 400", "time", "intensity");
	// Plot plot("wxt size 640, 400", "", "");
	plot.PlotLinePoint(gp_x, gp_y, gp_x_point, gp_y_point, line_title, point_title, colors);
}

void IProcess::PlotIntensityFunction(const Sequence& data, const unsigned& dim_id)
{
	double delta = 0.01;

	const double& Tc = data.GetTimeWindow();

	unsigned num_points = unsigned(Tc / delta);

	Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(num_points, 0, Tc);

	Eigen::VectorXd y = Eigen::VectorXd::Zero(num_points);

	Eigen::VectorXd intensity_dim(num_dims_);

	for(unsigned i = 0; i < num_points; ++ i)
	{
		Intensity(t(i), data, intensity_dim);
		y(i) = intensity_dim(dim_id);
	}
	
	const std::vector<Event>& events = data.GetEvents();
	std::vector<double> gp_x_point, gp_y_point;

	for(std::vector<Event>::const_iterator i_event = events.begin(); i_event != events.end(); ++ i_event)
	{
		if(i_event->DimentionID == dim_id)
		{
			gp_x_point.push_back(i_event->time);
			gp_y_point.push_back(-0.2);	
		}
	}

	std::vector<double> gp_x(num_points, 0);
	std::vector<double> gp_y(num_points, 0);

	for(unsigned i = 0; i < num_points; ++ i)
	{
		gp_x[i] = t(i);
		gp_y[i] = y(i);
	}

	Plot plot("wxt size 640, 400", "time", "intensity");
	plot.PlotLinePoint(gp_x, gp_y, gp_x_point, gp_y_point, "Intensity Function", "events");
}