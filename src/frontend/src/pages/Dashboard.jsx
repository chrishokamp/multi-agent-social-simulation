import React, { useState, useEffect } from 'react';
import { useParams, useLocation, useNavigate, Link } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
import Navbar from './components/Navbar';
import api from '/src/services/apiService';

import { TbArrowBackUp } from 'react-icons/tb';
import { IoEyeOutline, IoEyeOffOutline } from 'react-icons/io5';
import { IoMdSave } from 'react-icons/io';

const Dashboard = () => {
  const [simulationData, setSimulationData] = useState(null);
  const [simulationId, setSimulationId] = useState('');
  const [isOpeningScreenVisible, setIsOpeningScreenVisible] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // State for simulation catalog
  const [simulationCatalog, setSimulationCatalog] = useState([]);
  const [catalogLoading, setCatalogLoading] = useState(false);

  // For navigation
  const navigate = useNavigate();

  // Enhanced visualization state
  const [chartConfig, setChartConfig] = useState({
    type: 'bar', // 'bar', 'line', 'scatter', 'area', 'pie'
    xAxis: '', // Variable name for X-axis
    yAxis: '', // Variable name for Y-axis
    showTrendline: false,
    showLegend: true,
    showDataLabels: false,
  });

  // Advanced filtering and comparison
  const [filterOptions, setFilterOptions] = useState({
    enabled: false,
    variable: '',
    operator: '>',
    value: 0,
  });

  const [comparison, setComparison] = useState({
    enabled: false,
    variable: '',
    runs: [],
  });

  // Display mode state
  const [advancedMode, setAdvancedMode] = useState(false);

  // Track available variables dynamically
  const [availableVariables, setAvailableVariables] = useState([]);
  const [secondaryYAxis, setSecondaryYAxis] = useState({
    enabled: false,
    variable: '',
  });

  // UI state for collapsible sections
  const [collapsibleSections, setCollapsibleSections] = useState({
    chartConfig: true,
    statistics: true,
    variables: true,
    dataTable: false,
  });

  // Toggle collapsible section
  const toggleSection = (section) => {
    setCollapsibleSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  // Get simulationId from URL params if available
  const params = useParams();

  // Detect if we should show the opening screen
  useEffect(() => {
    if (params.simulationId) {
      // If simulationId exists in URL params, use it and don't show the opening screen
      setSimulationId(params.simulationId);
      setIsOpeningScreenVisible(false);
    } else {
      // If no simulationId in URL, show the opening screen
      setIsOpeningScreenVisible(true);
      fetchSimulationCatalog();
      setLoading(false);
    }
  }, [params.simulationId]);

  // Fetch data from API when simulationId changes
  useEffect(() => {
    const fetchSimulationData = async () => {
      if (!simulationId) return;

      setLoading(true);
      try {
        const data = await api.getSimulationOutput(simulationId);
        setSimulationData(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching simulation data:', error);
        setError('Failed to load simulation data. Please check the ID and try again.');
        setLoading(false);
      }
    };

    fetchSimulationData();
  }, [simulationId]);

  // Extract variables when simulation data changes
  useEffect(() => {
    if (!simulationData) return;

    // Extract unique output variable names from all runs
    const varSet = new Set();

    // Add run index as an available variable
    varSet.add('run_index');

    // Add number of messages as an available variable
    varSet.add('num_messages');

    // Add all output variables
    simulationData.runs.forEach((run) => {
      run.output_variables?.forEach((variable) => {
        varSet.add(variable.name);
      });
    });

    // Convert to array for dropdown selection
    const variables = Array.from(varSet);
    setAvailableVariables(variables);

    // Set default axes if not already set
    if (!chartConfig.xAxis && variables.length > 0) {
      setChartConfig((prev) => ({
        ...prev,
        xAxis: 'run_index',
        yAxis: variables.find((v) => v !== 'run_index') || variables[0],
      }));
    }
  }, [simulationData]);

  // Handle Simulation ID input change
  const handleSimulationIdChange = (event) => {
    setSimulationId(event.target.value);
  };

  const handleDownloadConfig = async () => {
    try {
      const cfg = await api.getSimulationConfig(simulationId);
      const blob = new Blob([JSON.stringify(cfg, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `simulation_${simulationId}.json`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(`Failed to download config: ${err.message}`);
    }
  };


  // Submit simulation ID to view dashboard
  const handleViewDashboard = async () => {
    if (!simulationId) {
      setError('Simulation ID cannot be empty.');
      return;
    }

    try {
      // Check if the simulation exists
      const response = await fetch(`http://localhost:5000/sim/results?id=${simulationId}`);

      if (!response.ok) {
        throw new Error('Simulation not found.');
      }

      // If simulation exists, hide the opening screen and update URL
      setError('');
      setIsOpeningScreenVisible(false);

      // Update the URL without refreshing the page (for bookmarking purposes)
      window.history.pushState({}, '', `/dashboard/${simulationId}`);
    } catch (error) {
      setError('Simulation ID does not exist. Please check the ID and try again.');
    }
  };

  // Get data for a specific variable across all runs
  const getDataForVariable = (variableName) => {
    if (!simulationData || !variableName) return [];

    if (variableName === 'run_index') {
      return simulationData.runs.map((_, index) => index + 1);
    }

    if (variableName === 'num_messages') {
      return simulationData.runs.map((run) => Number(run.num_messages || 0));
    }

    return simulationData.runs.map((run) => {
      const found = run.output_variables?.find((v) => v.name === variableName);
      // Try to convert to number if possible, otherwise return the string value
      const value = found ? found.value : null;
      return !isNaN(Number(value)) ? Number(value) : value;
    });
  };

  // Check if data is numeric
  const isDataNumeric = (data) => {
    return data.every((val) => val !== null && !isNaN(Number(val)));
  };

  // Apply filters to data
  const applyFilters = (data) => {
    if (!filterOptions.enabled || !filterOptions.variable || !filterOptions.operator) {
      return data;
    }

    const filterValues = getDataForVariable(filterOptions.variable);
    const filterValue = Number(filterOptions.value);

    return data.filter((_, index) => {
      const val = Number(filterValues[index]);

      if (isNaN(val)) return true;

      switch (filterOptions.operator) {
        case '>':
          return val > filterValue;
        case '>=':
          return val >= filterValue;
        case '<':
          return val < filterValue;
        case '<=':
          return val <= filterValue;
        case '==':
          return val === filterValue;
        case '!=':
          return val !== filterValue;
        default:
          return true;
      }
    });
  };

  // Generate chart options based on current configuration
  const generateChartOptions = () => {
    if (!simulationData) return {};

    // Get raw data
    let xData = getDataForVariable(chartConfig.xAxis);
    let yData = getDataForVariable(chartConfig.yAxis);

    // Create data points
    let dataPoints = xData.map((x, index) => ({
      index,
      x,
      y: yData[index],
      runIndex: index + 1,
    }));

    // New code for bar charts
    const isBarChart = chartConfig.type === 'bar';
    const barCategories = dataPoints.map((p) => String(p.x));
    const barValues = dataPoints.map((p) => p.y);

    // Apply filters if enabled
    if (filterOptions.enabled) {
      dataPoints = applyFilters(dataPoints);
    }

    // Re-extract x and y values after filtering
    xData = dataPoints.map((p) => p.x);
    yData = dataPoints.map((p) => p.y);

    // Determine if data is numeric
    const xNumeric = isDataNumeric(xData);
    const yNumeric = isDataNumeric(yData);

    // Generate series
    let series = [];

    // Define chart colors that match the site's palette
    const primaryColor = 'hsl(var(--accent-pro-200))';
    const secondaryColor = 'hsl(var(--accent-secondary-100))';
    const accentColor = 'hsl(var(--accent-pro-100))';

    // Handle comparison mode
    if (comparison.enabled && comparison.variable && comparison.runs.length > 0) {
      // Main series
      series.push({
        name: chartConfig.yAxis,
        type: chartConfig.type === 'area' ? 'line' : chartConfig.type,
        barMaxWidth: chartConfig.type === 'bar' ? '60%' : undefined,
        areaStyle: chartConfig.type === 'area' ? { color: primaryColor, opacity: 0.3 } : undefined,
        itemStyle: { color: primaryColor },
        data: dataPoints.map((point) => [point.x, point.y]),
        emphasis: { focus: 'series' },
        label: {
          show: chartConfig.showDataLabels,
          position: 'top',
          textStyle: { color: '#fff' },
        },
        markLine: chartConfig.showTrendline
          ? {
              silent: true,
              lineStyle: {
                color: '#fff',
                width: 1,
                opacity: 0.5,
                type: 'dashed',
              },
              data: [{ type: 'average', name: 'Avg' }],
            }
          : undefined,
      });

      // Comparison series
      const compareData = getDataForVariable(comparison.variable);

      comparison.runs.forEach((runIndex, idx) => {
        const actualIndex = runIndex - 1;
        // Use different shades of the accent color for each comparison
        const comparisonColor = `rgba(139, 92, 246, ${0.5 + idx * 0.1})`; // violet-500 with varying opacity

        series.push({
          name: `Run ${runIndex} - ${comparison.variable}`,
          type: 'line',
          itemStyle: { color: comparisonColor },
          markPoint: {
            symbolSize: 15,
            itemStyle: { color: comparisonColor },
            data: [
              {
                name: `Run ${runIndex}`,
                xAxis: xData.includes(xData[actualIndex]) ? xData[actualIndex] : null,
                yAxis: compareData[actualIndex],
              },
            ],
          },
          data: [
            [
              xData.includes(xData[actualIndex]) ? xData[actualIndex] : null,
              compareData[actualIndex],
            ],
          ],
        });
      });
    }
    // Handle secondary Y-axis
    else if (secondaryYAxis.enabled && secondaryYAxis.variable) {
      // Main Y-axis series
      series.push({
        name: chartConfig.yAxis,
        type: chartConfig.type === 'area' ? 'line' : chartConfig.type,
        barMaxWidth: chartConfig.type === 'bar' ? '60%' : undefined,
        areaStyle: chartConfig.type === 'area' ? { color: primaryColor, opacity: 0.3 } : undefined,
        itemStyle: { color: primaryColor },
        data: dataPoints.map((point) => [point.x, point.y]),
        emphasis: { focus: 'series' },
        label: {
          show: chartConfig.showDataLabels,
          position: 'top',
          textStyle: { color: '#fff' },
        },
        markLine: chartConfig.showTrendline
          ? {
              silent: true,
              lineStyle: {
                color: '#fff',
                width: 1,
                opacity: 0.5,
                type: 'dashed',
              },
              data: [{ type: 'average', name: 'Avg' }],
            }
          : undefined,
      });

      // Secondary Y-axis series
      const secondaryYData = getDataForVariable(secondaryYAxis.variable);
      series.push({
        name: secondaryYAxis.variable,
        type: 'line',
        yAxisIndex: 1,
        itemStyle: { color: secondaryColor },
        data: dataPoints.map((point) => [point.x, secondaryYData[point.index]]),
        emphasis: { focus: 'series' },
      });
    }
    // Standard single series
    else {
      series.push({
        name: chartConfig.yAxis,
        type: chartConfig.type === 'area' ? 'line' : chartConfig.type,
        barMaxWidth: chartConfig.type === 'bar' ? '60%' : undefined,
        areaStyle: chartConfig.type === 'area' ? { color: primaryColor, opacity: 0.3 } : undefined,
        itemStyle: { color: primaryColor },
        smooth: chartConfig.type === 'line' || chartConfig.type === 'area',
        data: isBarChart ? barValues : dataPoints.map((point) => [point.x, point.y]),
        emphasis: { focus: 'series' },
        label: {
          show: chartConfig.showDataLabels,
          position: 'top',
          textStyle: { color: '#fff' },
        },
        markLine: chartConfig.showTrendline
          ? {
              silent: true,
              lineStyle: {
                color: '#fff',
                width: 1,
                opacity: 0.5,
                type: 'dashed',
              },
              data: [{ type: 'average', name: 'Avg' }],
            }
          : undefined,
      });
    }

    // Build chart options
    const chartOptions = {
      backgroundColor: 'transparent',
      title: {
        text: `${chartConfig.yAxis} vs ${chartConfig.xAxis}`,
        left: 'center',
        top: 10,
        textStyle: { color: 'hsl(var(--oncolor-100))' },
      },
      grid: {
        left: '5%', // Increased to prevent left overflow
        right: secondaryYAxis.enabled ? '10%' : '5%', // Increased for secondary axis
        bottom: '15%', // Increased to prevent bottom overflow
        top: '18%', // Increased to prevent top overflow
        containLabel: true, // Ensures labels are contained within the grid
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
          label: {
            backgroundColor: 'hsl(var(--accent-pro-000))',
          },
        },
        formatter: function (params) {
          let result = '';
          params.forEach((param) => {
            // For bar charts, param.value is a single number
            // For line/scatter, param.value is an [x,y] array
            const val = Array.isArray(param.value) ? param.value[1] : param.value;
            const runIndex = param.dataIndex + 1;
            result += `Run ${runIndex}<br/>${param.seriesName}: ${val}<br/><br/>`;
          });
          return result;
        },
        textStyle: {
          color: 'hsl(var(--oncolor-100))',
        },
        backgroundColor: 'hsl(var(--accent-pro-900))',
        borderColor: 'hsl(var(--accent-pro-200))',
        borderWidth: 1,
      },
      legend: {
        data: series.map((s) => s.name),
        top: 40,
        left: 'center',
        textStyle: { color: 'hsl(var(--oncolor-100))' },
        show: chartConfig.showLegend,
        type: 'scroll', // Enable scrolling for many items
        pageButtonPosition: 'end', // Position of page buttons
        pageButtonItemGap: 5, // Gap between page buttons
        pageButtonGap: 5, // Gap between page buttons and page info
        pageIconColor: 'hsl(var(--accent-pro-200))',
        pageIconInactiveColor: 'hsl(var(--accent-pro-900))',
        pageIconSize: 12, // Size of page buttons
        pageTextStyle: {
          color: 'hsl(var(--accent-pro-200))',
        },
      },
      toolbox: {
        feature: {
          saveAsImage: {
            icon: 'path://M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l2.007-7.454c.158-.474.457-.85.85-1.093A2.25 2.25 0 018 6h8v9zm-1-7a1 1 0 10-2 0 1 1 0 002 0z',
            title: 'Save as Image',
            iconStyle: {
              color: 'hsl(var(--accent-pro-200))',
              borderColor: 'hsl(var(--oncolor-100))',
              borderWidth: 1,
            },
            emphasis: {
              iconStyle: {
                color: 'hsl(var(--accent-pro-100))',
              },
            },
          },
        },
        right: 20,
        top: 20,
        itemSize: 20, // Larger icon
        itemGap: 10,
      },
      xAxis: {
        type: isBarChart ? 'category' : xNumeric ? 'value' : 'category',
        data: isBarChart ? barCategories : undefined,
        boundaryGap: isBarChart, // true for bar charts, false otherwise
        name: chartConfig.xAxis,
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: {
          color: 'hsl(var(--oncolor-100))',
        },
        axisLabel: {
          color: 'hsl(var(--oncolor-100))',
          formatter: xNumeric ? (value) => value : null,
          margin: 15, // Extra margin to prevent cutoff
          rotate: xData.length > 10 ? 45 : 0, // Rotate labels if there are many points
        },
        axisLine: {
          lineStyle: {
            color: 'hsl(var(--accent-pro-000))',
          },
        },
        axisTick: {
          lineStyle: {
            color: 'hsl(var(--accent-pro-000))',
          },
        },
        splitLine: {
          lineStyle: {
            color: 'hsl(var(--accent-pro-900))',
            opacity: 0.3,
          },
        },
      },
      yAxis: [
        {
          type: 'value',
          name: chartConfig.yAxis,
          nameLocation: 'middle',
          nameGap: 50,
          nameTextStyle: {
            color: 'hsl(var(--oncolor-100))',
          },
          axisLabel: {
            color: 'hsl(var(--oncolor-100))',
            margin: 15, // Extra margin to prevent cutoff
          },
          axisLine: {
            lineStyle: {
              color: 'hsl(var(--accent-pro-000))',
            },
          },
          axisTick: {
            lineStyle: {
              color: 'hsl(var(--accent-pro-000))',
            },
          },
          splitLine: {
            lineStyle: {
              color: 'hsl(var(--accent-pro-900))',
              opacity: 0.3,
            },
          },
          scale: true, // Better scale for numeric data to prevent edge clipping
          min: 0,
        },
        secondaryYAxis.enabled
          ? {
              type: 'value',
              name: secondaryYAxis.variable,
              nameLocation: 'middle',
              nameGap: 50,
              nameTextStyle: {
                color: 'hsl(var(--accent-secondary-100))',
              },
              axisLabel: {
                color: 'hsl(var(--accent-secondary-100))',
                margin: 15, // Extra margin to prevent cutoff
              },
              axisLine: {
                lineStyle: {
                  color: 'hsl(var(--accent-secondary-100))',
                },
              },
              axisTick: {
                lineStyle: {
                  color: 'hsl(var(--accent-secondary-100))',
                },
              },
              splitLine: {
                show: false, // Don't show secondary grid lines to reduce clutter
              },
              position: 'right',
              scale: true, // Better scale for numeric data to prevent edge clipping
              min: 0,
            }
          : undefined,
      ].filter(Boolean),
      series: series,
    };

    return chartOptions;
  };

  // Compute statistical summaries
  const computeStats = (variableName) => {
    const values = getDataForVariable(variableName);

    if (!values.length || !isDataNumeric(values)) return null;

    const nums = values.filter((val) => val !== null).map(Number);
    if (nums.length === 0) return null;

    const sum = nums.reduce((acc, num) => acc + num, 0);
    const mean = sum / nums.length;
    const sortedNums = [...nums].sort((a, b) => a - b);
    const median =
      sortedNums.length % 2 === 0
        ? (sortedNums[sortedNums.length / 2 - 1] + sortedNums[sortedNums.length / 2]) / 2
        : sortedNums[Math.floor(sortedNums.length / 2)];

    // Calculate standard deviation
    const squaredDiffs = nums.map((num) => Math.pow(num - mean, 2));
    const variance = squaredDiffs.reduce((acc, val) => acc + val, 0) / nums.length;
    const stdDev = Math.sqrt(variance);

    return {
      count: nums.length,
      mean: mean.toFixed(2),
      median: median.toFixed(2),
      stdDev: stdDev.toFixed(2),
      min: Math.min(...nums),
      max: Math.max(...nums),
    };
  };

  // Toggle between basic and advanced mode
  const toggleAdvancedMode = () => {
    setAdvancedMode(!advancedMode);
    // Reset comparison and filtering when toggling modes
    if (!advancedMode) {
      setComparison({ enabled: false, variable: '', runs: [] });
      setFilterOptions({ enabled: false, variable: '', operator: '>', value: 0 });
      setSecondaryYAxis({ enabled: false, variable: '' });
    }
  };

  // Generate message chart data
  const generateMessageChartOption = () => {
    if (!simulationData) return {};

    const messageLabels = simulationData.runs.map((_, index) => `Run ${index + 1}`);
    const messagesData = simulationData.runs.map((run) => Number(run.num_messages || 0));

    return {
      title: {
        text: 'Messages per Run',
        left: 'center',
        textStyle: { color: 'hsl(var(--oncolor-100))' },
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'hsl(var(--accent-pro-900))',
        borderColor: 'hsl(var(--accent-pro-200))',
        borderWidth: 1,
        textStyle: {
          color: 'hsl(var(--oncolor-100))',
        },
      },
      grid: {
        left: '5%',
        right: '5%',
        bottom: '15%',
        top: '18%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: messageLabels,
        axisLabel: {
          color: 'hsl(var(--oncolor-100))',
          margin: 10,
        },
        axisLine: {
          lineStyle: {
            color: 'hsl(var(--accent-pro-000))',
          },
        },
        axisTick: {
          lineStyle: {
            color: 'hsl(var(--accent-pro-000))',
          },
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          color: 'hsl(var(--oncolor-100))',
          margin: 10,
        },
        axisLine: {
          lineStyle: {
            color: 'hsl(var(--accent-pro-000))',
          },
        },
        axisTick: {
          lineStyle: {
            color: 'hsl(var(--accent-pro-000))',
          },
        },
        splitLine: {
          lineStyle: {
            color: 'hsl(var(--accent-pro-900))',
            opacity: 0.3,
          },
        },
        scale: true, // Better scale for numeric data
      },
      toolbox: {
        feature: {
          saveAsImage: {
            icon: 'path://M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l2.007-7.454c.158-.474.457-.85.85-1.093A2.25 2.25 0 018 6h8v9zm-1-7a1 1 0 10-2 0 1 1 0 002 0z',
            title: 'Save as Image',
            iconStyle: {
              color: 'hsl(var(--accent-pro-200))',
              borderColor: 'hsl(var(--oncolor-100))',
              borderWidth: 1,
            },
            emphasis: {
              iconStyle: {
                color: 'hsl(var(--accent-pro-100))',
              },
            },
          },
        },
        right: 20,
        top: 20,
        itemSize: 20, // Larger icon
        itemGap: 10,
      },
      series: [
        {
          type: 'line',
          data: messagesData,
          smooth: true,
          itemStyle: {
            color: 'hsl(var(--accent-secondary-100))',
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'hsl(var(--accent-secondary-100) / 0.5)' },
                { offset: 1, color: 'hsl(var(--accent-secondary-100) / 0)' },
              ],
            },
          },
        },
      ],
      backgroundColor: 'transparent',
    };
  };

  // Fetch simulation catalog
  const fetchSimulationCatalog = async () => {
    setCatalogLoading(true);
    try {
      const catalog = await api.getSimulationsCatalog();
      setSimulationCatalog(catalog);
      setCatalogLoading(false);
    } catch (error) {
      console.error('Error fetching simulation catalog:', error);
      setError('Failed to load simulation catalog. Please try again later.');
      setCatalogLoading(false);
    }
  };

  // Handle direct simulation selection (when clicking a card)
  const handleDirectSimulationSelect = (id) => {
    navigate(`/dashboard/${id}`);
  };

  if (loading && !isOpeningScreenVisible) {
    return (
      <div className="w-full min-h-screen bg-transparent flex flex-col justify-center items-center">
        <Navbar />
        <div className=" text-xl mt-20">Loading dashboard data...</div>
      </div>
    );
  }

  // Render opening screen if no simulation ID is provided
  if (isOpeningScreenVisible) {
    return (
      <div className="w-full min-h-screen">
        <Navbar />
        <div className="fixed inset-0 flex items-center justify-center">
          <div className=" container backdrop-blur-3xl p-10 w-96 md:w-1/2 lg:w-1/3 min-h-[300px] flex flex-col justify-center rounded-xl shadow-card text-center">
            <h2 className="text-2xl md:text-3xl font-bold">Select a Simulation</h2>
            <p className="text-lg">
              Choose the simulation that you want to analyse.
            </p>

            {/* Simulation List as Cards */}
            <div className="mt-6">
              {simulationCatalog.map((sim) => (
                <div
                  key={sim.simulation_id}
                  onClick={() => setSimulationId(sim.simulation_id)}
                  className={`container p-3 my-2 flex justify-between items-center border rounded-lg text-left cursor-pointer transition-colors duration-200 ${
                    simulationId === sim.simulation_id
                      ? ' border'
                      : 'border'
                  }`}
                >
                  <p className="">{sim.name || `Simulation ${sim.simulation_id}`}</p>
                  {simulationId === sim.simulation_id && (
                    <span className="text-emerald-500 text-xl">&#x2713;</span> // Checkmark icon
                  )}
                </div>
              ))}
            </div>

            {/* Error Message */}
            {error && <p className="danger">{error}</p>}

            {/* View Dashboard Button */}
            <div className="mt-6">
              <button
                onClick={() => handleDirectSimulationSelect(simulationId)}
                disabled={!simulationId}
                className={`${
                  !simulationId
                    ? 'cursor-not-allowed'
                    : ' hover:shadow-button'
                }  px-6 py-3 rounded-full transition-colors duration-200 cursor-pointer`}
              >
                View Dashboard
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // If there's no simulation data, show an error
  if (!simulationData && !loading) {
    return (
      <div className="w-full min-h-screen bg-transparent p-4">
        <Navbar />
        <div className="danger">
          Error loading simulation data. Please check the simulation ID and try again.
        </div>
      </div>
    );
  }

  // Render main dashboard
  return (
    <div className="w-full min-h-screen p-4 bg-transparent">
      <Navbar />
      <div className="flex justify-between items-center mb-4 mt-22">
        <h1 className="text-2xl  font-bold">Simulation Dashboard</h1>

        {/* action buttons */}
        <div className="flex space-x-2">
          <Link
            to="/simulations"
            className="px-4 py-2 hover:shadow-button  rounded-lg transition-colors flex items-center"
          >
            <TbArrowBackUp className="mr-2 h-5 w-5" />
            Catalog
          </Link>

          <button
            onClick={handleDownloadConfig}
            className="px-4 py-2 hover:shadow-button  rounded-lg transition-colors flex items-center"
          >
            <IoMdSave className="mr-2 h-5 w-5" />
            Download&nbsp;JSON
          </button>
        </div>
      </div>
    
      {/* Simulation Info Panel */}
      <div className="">
        <div className="p-2 rounded mb-4">
          <p>
            <span className="font-bold">Simulation ID: </span>
            {simulationId}
          </p>
          <p>
            <span className="font-bold">Agents: </span>
            {Array.from(
              new Set(
                simulationData.runs
                  .flatMap((run) => run.messages?.map((msg) => msg.agent) || [])
                  .filter((agent) => agent !== 'InformationReturnAgent')
              )
            ).join(', ')}
          </p>
          <p>
            <span className="font-bold">Total Runs:</span> {simulationData.runs.length}
          </p>
        </div>
      </div>

      {/* Messages per Run Chart */}
      <div className="mb-6">
        <ReactECharts option={generateMessageChartOption()} className="h-80 w-full" />
      </div>
      
      {/* Mode Selector */}
      <div className="mb-6 flex space-x-4">
        <button
          onClick={toggleAdvancedMode}
          className={`px-4 py-2 rounded-md ${
            !advancedMode
              ? ' '
              : ' border  '
          }`}
        >
          Basic Mode
        </button>
        <button
          onClick={toggleAdvancedMode}
          className={`px-4 py-2 rounded-md ${
            advancedMode
              ? ''
              : ' border '
          }`}
        >
          Advanced Mode
        </button>
      </div>

      {/* Chart Configuration Panel */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4 p-2 bg-violet-800/20 rounded">
          <h2 className="text-xl  font-semibold">Chart Configuration</h2>
          <button
            onClick={() => toggleSection('chartConfig')}
            className="p-2  rounded-md transition-colors"
          >
            {collapsibleSections.chartConfig ? (
              <IoEyeOffOutline size={20} />
            ) : (
              <IoEyeOutline size={20} />
            )}
          </button>
        </div>

        {collapsibleSections.chartConfig && (
          <div className="p-4  rounded-lg border">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 ">
              {/* Chart Type Selection */}
              <div>
                <label className="block mb-2">Chart Type</label>
                <select
                  value={chartConfig.type}
                  onChange={(e) => setChartConfig({ ...chartConfig, type: e.target.value })}
                  className="w-full p-2 border rounded  "
                >
                  <option value="bar">Bar Chart</option>
                  <option value="line">Line Chart</option>
                  <option value="scatter">Scatter Plot</option>
                  <option value="area">Area Chart</option>
                  <option value="pie">Pie Chart</option>
                </select>
              </div>

              {/* X-Axis Selection */}
              <div>
                <label className="block mb-2">X-Axis Variable</label>
                <select
                  value={chartConfig.xAxis}
                  onChange={(e) => setChartConfig({ ...chartConfig, xAxis: e.target.value })}
                  className="w-full p-2 border rounded  "
                >
                  <option value="">--Select--</option>
                  {availableVariables.map((variable) => (
                    <option key={`x-${variable}`} value={variable}>
                      {variable}
                    </option>
                  ))}
                </select>
              </div>

              {/* Y-Axis Selection */}
              <div>
                <label className="block mb-2">Y-Axis Variable</label>
                <select
                  value={chartConfig.yAxis}
                  onChange={(e) => setChartConfig({ ...chartConfig, yAxis: e.target.value })}
                  className="w-full p-2 border rounded  "
                >
                  <option value="">--Select--</option>
                  {availableVariables.map((variable) => (
                    <option key={`y-${variable}`} value={variable}>
                      {variable}
                    </option>
                  ))}
                </select>
              </div>

              {/* Visualization Options */}
              <div className="flex flex-col justify-between">
                <label className="block mb-2">Visualization Options</label>
                <div className="flex flex-col space-y-2">
                  <label className="inline-flex items-center">
                    <input
                      type="checkbox"
                      checked={chartConfig.showTrendline}
                      onChange={(e) =>
                        setChartConfig({ ...chartConfig, showTrendline: e.target.checked })
                      }
                      className="mr-2"
                    />
                    Show Trendline
                  </label>
                  <label className="inline-flex items-center">
                    <input
                      type="checkbox"
                      checked={chartConfig.showLegend}
                      onChange={(e) =>
                        setChartConfig({ ...chartConfig, showLegend: e.target.checked })
                      }
                      className="mr-2"
                    />
                    Show Legend
                  </label>
                  <label className="inline-flex items-center">
                    <input
                      type="checkbox"
                      checked={chartConfig.showDataLabels}
                      onChange={(e) =>
                        setChartConfig({ ...chartConfig, showDataLabels: e.target.checked })
                      }
                      className="mr-2"
                    />
                    Show Data Labels
                  </label>
                </div>
              </div>
            </div>

            {/* Advanced Options (only in advanced mode) */}
            {advancedMode && (
              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 ">
                {/* Secondary Y-Axis */}
                <div className="p-3 border  rounded ">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-medium ">Secondary Y-Axis</h3>
                    <label className="inline-flex items-center">
                      <input
                        type="checkbox"
                        checked={secondaryYAxis.enabled}
                        onChange={(e) =>
                          setSecondaryYAxis({ ...secondaryYAxis, enabled: e.target.checked })
                        }
                        className="mr-2"
                      />
                      Enable
                    </label>
                  </div>

                  {secondaryYAxis.enabled && (
                    <div>
                      <label className="block mb-2">Select Variable</label>
                      <select
                        value={secondaryYAxis.variable}
                        onChange={(e) =>
                          setSecondaryYAxis({ ...secondaryYAxis, variable: e.target.value })
                        }
                        className="w-full p-2 border rounded  "
                      >
                        <option value="">--Select--</option>
                        {availableVariables.map((variable) => (
                          <option key={`secondary-${variable}`} value={variable}>
                            {variable}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                </div>

                {/* Data Filtering */}
                <div className="p-3 border  rounded ">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-medium ">Data Filtering</h3>
                    <label className="inline-flex items-center">
                      <input
                        type="checkbox"
                        checked={filterOptions.enabled}
                        onChange={(e) =>
                          setFilterOptions({ ...filterOptions, enabled: e.target.checked })
                        }
                        className="mr-2"
                      />
                      Enable
                    </label>
                  </div>

                  {filterOptions.enabled && (
                    <div className="grid grid-cols-3 gap-2">
                      <div>
                        <label className="block mb-2">Variable</label>
                        <select
                          value={filterOptions.variable}
                          onChange={(e) =>
                            setFilterOptions({ ...filterOptions, variable: e.target.value })
                          }
                          className="w-full p-2 border rounded  "
                        >
                          <option value="">--Select--</option>
                          {availableVariables
                            .filter((variable) => isDataNumeric(getDataForVariable(variable)))
                            .map((variable) => (
                              <option key={`filter-${variable}`} value={variable}>
                                {variable}
                              </option>
                            ))}
                        </select>
                      </div>

                      <div>
                        <label className="block mb-2">Operator</label>
                        <select
                          value={filterOptions.operator}
                          onChange={(e) =>
                            setFilterOptions({ ...filterOptions, operator: e.target.value })
                          }
                          className="w-full p-2 border rounded  "
                        >
                          <option value=">">Greater than (&gt;)</option>
                          <option value=">=">Greater than or equal (&gt;=)</option>
                          <option value="<">Less than (&lt;)</option>
                          <option value="<=">Less than or equal (&lt;=)</option>
                          <option value="==">Equal to (==)</option>
                          <option value="!=">Not equal to (!=)</option>
                        </select>
                      </div>

                      <div>
                        <label className="block mb-2">Value</label>
                        <input
                          type="number"
                          value={filterOptions.value}
                          onChange={(e) =>
                            setFilterOptions({ ...filterOptions, value: e.target.value })
                          }
                          className="w-full p-2 border rounded  "
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Run Comparison */}
                <div className="p-3 border  rounded  md:col-span-2">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-medium ">Run Comparison</h3>
                    <label className="inline-flex items-center">
                      <input
                        type="checkbox"
                        checked={comparison.enabled}
                        onChange={(e) =>
                          setComparison({ ...comparison, enabled: e.target.checked })
                        }
                        className="mr-2"
                      />
                      Enable
                    </label>
                  </div>

                  {comparison.enabled && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block mb-2">Comparison Variable</label>
                        <select
                          value={comparison.variable}
                          onChange={(e) =>
                            setComparison({ ...comparison, variable: e.target.value })
                          }
                          className="w-full p-2 border rounded  "
                        >
                          <option value="">--Select--</option>
                          {availableVariables.map((variable) => (
                            <option key={`comp-${variable}`} value={variable}>
                              {variable}
                            </option>
                          ))}
                        </select>
                      </div>

                      <div>
                        <label className="block mb-2">Select Runs to Compare</label>
                        <div className="flex flex-wrap gap-2">
                          {simulationData.runs.map((_, index) => (
                            <button
                              key={`run-${index + 1}`}
                              onClick={() => {
                                const runIndex = index + 1;
                                setComparison((prev) => {
                                  const isSelected = prev.runs.includes(runIndex);
                                  return {
                                    ...prev,
                                    runs: isSelected
                                      ? prev.runs.filter((r) => r !== runIndex)
                                      : [...prev.runs, runIndex],
                                  };
                                });
                              }}
                              className={`px-2 py-1 rounded text-sm ${
                                comparison.runs.includes(index + 1)
                                  ? ' '
                                  : ' '
                              }`}
                            >
                              Run {index + 1}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Main Visualization */}
      <div className="mb-6">
        {chartConfig.xAxis && chartConfig.yAxis ? (
          <div className="bg-violet-950/30 p-4 rounded-lg border border-violet-500/30">
            <ReactECharts
              option={generateChartOptions()}
              style={{ height: '500px', width: '100%' }}
              className="bg-transparent"
              opts={{ renderer: 'canvas' }}
            />
          </div>
        ) : (
          <div className="empty-panel">
            <p className="empty-panel-title">Please select variables for X and Y axes to visualize data.</p>
            <p className="empty-panel-desc">
              Use the Chart Configuration panel above to customize your visualization.
            </p>
          </div>
        )}
      </div>

      {/* Statistics Panel */}
      {chartConfig.yAxis && isDataNumeric(getDataForVariable(chartConfig.yAxis)) && (
        <div className="mb-6">
          <div className="panel-header">
            <h2 className="panel-title">Statistics for {chartConfig.yAxis}</h2>
            <button
              onClick={() => toggleSection('statistics')}
              className="panel-toggle"
            >
              {collapsibleSections.statistics ? (
                <IoEyeOffOutline size={20} />
              ) : (
                <IoEyeOutline size={20} />
              )}
            </button>
          </div>

          {collapsibleSections.statistics && (
            <div className="panel-body">
              <div className="stats-grid">
                {Object.entries(computeStats(chartConfig.yAxis) || {}).map(([stat, value]) => (
                  <div
                    key={stat}
                    className="stat-card"
                  >
                    <h3 className="stat-label">
                      {stat.charAt(0).toUpperCase() + stat.slice(1)}
                    </h3>
                    <p className="stat-value">{value}</p>
                  </div>
                ))}
              </div>

              {secondaryYAxis.enabled && secondaryYAxis.variable && (
                <div className="mt-4">
                  <h3 className="stat-secondary-title">
                    Statistics for {secondaryYAxis.variable} (Secondary Y-Axis)
                  </h3>
                  <div className="stats-grid">
                    {Object.entries(computeStats(secondaryYAxis.variable) || {}).map(
                      ([stat, value]) => (
                        <div
                          key={stat}
                          className="stat-card secondary"
                        >
                          <h3 className="text-teal-300 font-medium">
                            {stat.charAt(0).toUpperCase() + stat.slice(1)}
                          </h3>
                          <p className=" text-xl font-semibold">{value}</p>
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* All Variables Summary */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4 p-2 rounded">
          <h2 className="text-xl font-semibold">All Variables Summary</h2>
          <button
            onClick={() => toggleSection('variables')}
            className="p-2rounded-md transition-colors"
          >
            {collapsibleSections.variables ? (
              <IoEyeOffOutline size={20} />
            ) : (
              <IoEyeOutline size={20} />
            )}
          </button>
        </div>

        {collapsibleSections.variables && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {availableVariables.map((variable) => {
              const values = getDataForVariable(variable);
              const numeric = isDataNumeric(values);
              const stats = numeric ? computeStats(variable) : null;

              return (
                <div
                  key={variable}
                  className=" border p-4 rounded shadow hover:shadow-md cursor-pointer transition-all duration-200"
                  onClick={() => setChartConfig({ ...chartConfig, yAxis: variable })}
                >
                  <h3 className="font-medium text-lg mb-2">{variable}</h3>
                  {numeric ? (
                    <div>
                      <p className="text-sm">Mean: {stats.mean}</p>
                      <p className="text-sm">
                        Range: {stats.min} - {stats.max}
                      </p>
                      <p className="text-sm">StdDev: {stats.stdDev}</p>
                    </div>
                  ) : (
                    <p className="text-sm">
                      {values.slice(0, 3).join(', ')}
                      {values.length > 3 ? '...' : ''}
                    </p>
                  )}
                  <div className="mt-2 text-xs">Click to set as Y-axis</div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Data Table */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4 p-2 rounded">
          <h2 className="text-xlfont-semibold">Data Table</h2>
          <button
            onClick={() => toggleSection('dataTable')}
            className="p-2"
          >
            {collapsibleSections.dataTable ? (
              <IoEyeOffOutline size={20} />
            ) : (
              <IoEyeOutline size={20} />
            )}
          </button>
        </div>

        {collapsibleSections.dataTable && (
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr >
                  <th >Run</th>
                  {availableVariables.map((variable) => (
                    <th
                      key={`header-${variable}`}
                      
                    >
                      {variable}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {simulationData.runs.map((run, index) => (
                  <tr key={`run-${index}`} >
                    <td >
                      {index + 1}
                    </td>
                    {availableVariables.map((variable) => {
                      let value;
                      if (variable === 'run_index') {
                        value = index + 1;
                      } else if (variable === 'num_messages') {
                        value = run.num_messages || 0;
                      } else {
                        const found = run.output_variables?.find((v) => v.name === variable);
                        value = found ? found.value : 'N/A';
                      }

                      return (
                        <td
                          key={`cell-${index}-${variable}`}
                          
                        >
                          {value}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
