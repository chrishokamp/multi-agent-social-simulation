import React, { useState, useEffect } from 'react';
import { useParams, useLocation, useNavigate, Link } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
import Navbar from './components/Navbar';
import ChatStream from './components/ChatStream';
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
      if (!found) return null;
      
      let value = found.value;
      
      // Handle nested utility objects
      if (variableName === 'utility' && typeof value === 'object' && value !== null) {
        // If it's a utility object like {"Buyer": 0.5, "Seller": 0.8}, 
        // return the sum or average of all agent utilities
        const utilityValues = Object.values(value).filter(v => typeof v === 'number');
        if (utilityValues.length > 0) {
          // Return average utility across all agents
          value = utilityValues.reduce((sum, v) => sum + v, 0) / utilityValues.length;
        }
      }
      
      // Handle cases where value might be an object with a numeric property
      if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        // Look for common numeric properties
        if ('value' in value) value = value.value;
        else if ('amount' in value) value = value.amount;
        else if ('score' in value) value = value.score;
        else if ('result' in value) value = value.result;
        // If still an object, try to get the first numeric value
        else {
          const numericValue = Object.values(value).find(v => typeof v === 'number');
          if (numericValue !== undefined) value = numericValue;
        }
      }
      
      // Try to convert to number if possible, otherwise return the value
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
    const barCategories = dataPoints.filter(p => p.x !== undefined && p.x !== null).map((p) => String(p.x));
    const barValues = dataPoints.filter(p => p.y !== undefined && p.y !== null).map((p) => p.y);

    // Apply filters if enabled
    if (filterOptions.enabled) {
      dataPoints = applyFilters(dataPoints);
    }

    // Re-extract x and y values after filtering
    xData = dataPoints.map((p) => p.x);
    yData = dataPoints.map((p) => p.y);
    
    // Filter out any undefined or null values to prevent animation errors
    dataPoints = dataPoints.filter(p => p.x !== undefined && p.x !== null && p.y !== undefined && p.y !== null);
    xData = xData.filter(x => x !== undefined && x !== null);
    yData = yData.filter(y => y !== undefined && y !== null);

    // Determine if data is numeric
    const xNumeric = isDataNumeric(xData);
    const yNumeric = isDataNumeric(yData);

    // Generate series
    let series = [];

    // Define chart colors - professional, muted palette
    const primaryColor = '#4B5563';  // gray-600
    const secondaryColor = '#7C3AED';  // violet-600
    const accentColor = '#2563EB';  // blue-600

    // Handle comparison mode
    if (comparison.enabled && comparison.variable && comparison.runs.length > 0) {
      // Main series
      series.push({
        name: chartConfig.yAxis,
        type: chartConfig.type === 'area' ? 'line' : chartConfig.type,
        barMaxWidth: chartConfig.type === 'bar' ? '60%' : undefined,
        areaStyle: chartConfig.type === 'area' ? { color: primaryColor, opacity: 0.15 } : undefined,
        itemStyle: { color: primaryColor },
        data: dataPoints.map((point) => [point.x, point.y]),
        emphasis: { focus: 'series' },
        label: {
          show: chartConfig.showDataLabels,
          position: 'top',
          textStyle: { 
            color: '#374151',  // gray-700
            fontSize: 11
          },
        },
        markLine: chartConfig.showTrendline
          ? {
              silent: true,
              lineStyle: {
                color: '#9CA3AF',  // gray-400
                width: 1,
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
        // Use different professional colors for each comparison
        const comparisonColors = ['#7C3AED', '#2563EB', '#059669', '#DC2626'];  // violet, blue, emerald, red
        const comparisonColor = comparisonColors[idx % comparisonColors.length];

        series.push({
          name: `Run ${runIndex} - ${comparison.variable}`,
          type: 'line',
          itemStyle: { color: comparisonColor },
          markPoint: {
            symbolSize: 15,
            itemStyle: { color: comparisonColor },
            data: actualIndex < xData.length && compareData[actualIndex] !== undefined ? [
              {
                name: `Run ${runIndex}`,
                xAxis: xData[actualIndex],
                yAxis: compareData[actualIndex],
              },
            ] : [],
          },
          data: actualIndex < xData.length && compareData[actualIndex] !== undefined ? [
            [
              xData[actualIndex],
              compareData[actualIndex],
            ],
          ] : [],
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
        areaStyle: chartConfig.type === 'area' ? { color: primaryColor, opacity: 0.15 } : undefined,
        itemStyle: { color: primaryColor },
        data: dataPoints.map((point) => [point.x, point.y]),
        emphasis: { focus: 'series' },
        label: {
          show: chartConfig.showDataLabels,
          position: 'top',
          textStyle: { 
            color: '#374151',  // gray-700
            fontSize: 11
          },
        },
        markLine: chartConfig.showTrendline
          ? {
              silent: true,
              lineStyle: {
                color: '#9CA3AF',  // gray-400
                width: 1,
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
        data: dataPoints.map((point) => {
          const yValue = secondaryYData[point.index];
          return yValue !== undefined && yValue !== null ? [point.x, yValue] : null;
        }).filter(d => d !== null),
        emphasis: { focus: 'series' },
      });
    }
    // Standard single series
    else {
      // Handle pie chart differently
      if (chartConfig.type === 'pie') {
        // For pie charts, we need name-value pairs
        const pieData = dataPoints
          .filter(p => p.x !== undefined && p.x !== null && p.y !== undefined && p.y !== null)
          .map((point) => ({
            name: String(point.x),
            value: Number(point.y) || 0
          }));
        
        series.push({
          name: chartConfig.yAxis,
          type: 'pie',
          radius: '50%',
          data: pieData,
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          },
          label: {
            show: chartConfig.showDataLabels,
            formatter: '{b}: {c} ({d}%)'
          }
        });
      } else {
        // Line, bar, scatter, area charts
        series.push({
          name: chartConfig.yAxis,
          type: chartConfig.type === 'area' ? 'line' : chartConfig.type,
          barMaxWidth: chartConfig.type === 'bar' ? '60%' : undefined,
          areaStyle: chartConfig.type === 'area' ? { color: primaryColor, opacity: 0.15 } : undefined,
          itemStyle: { color: primaryColor },
          smooth: chartConfig.type === 'line' || chartConfig.type === 'area',
          data: isBarChart ? barValues.filter(v => v !== undefined && v !== null) : dataPoints.map((point) => [point.x, point.y]),
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
    }

    // Build chart options
    const isPieChart = chartConfig.type === 'pie';
    const chartOptions = {
      backgroundColor: 'transparent',
      title: {
        text: isPieChart ? `${chartConfig.yAxis} by ${chartConfig.xAxis}` : `${chartConfig.yAxis} vs ${chartConfig.xAxis}`,
        left: 'left',
        top: 5,
        textStyle: { 
          color: '#1F2937',  // gray-800
          fontSize: 14,
          fontWeight: 600
        },
      },
      grid: isPieChart ? undefined : {
        left: '3%',
        right: secondaryYAxis.enabled ? '8%' : '3%',
        bottom: '12%',
        top: '12%',
        containLabel: true,
      },
      tooltip: {
        trigger: isPieChart ? 'item' : 'axis',
        axisPointer: isPieChart ? undefined : {
          type: 'cross',
          label: {
            backgroundColor: '#374151',  // gray-700
          },
        },
        formatter: isPieChart 
          ? '{a} <br/>{b}: {c} ({d}%)'
          : function (params) {
              let result = '';
              params.forEach((param) => {
                // For bar charts, param.value is a single number
                // For line/scatter, param.value is an [x,y] array
                const val = Array.isArray(param.value) ? param.value[1] : param.value;
                const runIndex = param.dataIndex + 1;
                result += `Run ${runIndex}<br/>${param.seriesName}: ${val}<br/>`;
              });
              return result;
            },
        textStyle: {
          color: '#F9FAFB',  // gray-50
          fontSize: 12
        },
        backgroundColor: '#1F2937',  // gray-800
        borderColor: '#374151',  // gray-700
        borderWidth: 1,
      },
      legend: {
        data: series.map((s) => s.name),
        top: 28,
        left: 'left',
        textStyle: { 
          color: '#4B5563',  // gray-600
          fontSize: 12
        },
        show: chartConfig.showLegend,
        type: 'scroll',
        pageButtonPosition: 'end',
        pageButtonItemGap: 3,
        pageButtonGap: 3,
        pageIconColor: '#6B7280',  // gray-500
        pageIconInactiveColor: '#D1D5DB',  // gray-300
        pageIconSize: 10,
        pageTextStyle: {
          color: '#6B7280',  // gray-500
          fontSize: 11
        },
      },
      toolbox: {
        feature: {
          saveAsImage: {
            icon: 'path://M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l2.007-7.454c.158-.474.457-.85.85-1.093A2.25 2.25 0 018 6h8v9zm-1-7a1 1 0 10-2 0 1 1 0 002 0z',
            title: 'Save',
            iconStyle: {
              color: '#6B7280',  // gray-500
              borderColor: '#E5E7EB',  // gray-200
              borderWidth: 1,
            },
            emphasis: {
              iconStyle: {
                color: '#4B5563',  // gray-600
              },
            },
          },
        },
        right: 10,
        top: 5,
        itemSize: 16,
        itemGap: 8,
      },
      xAxis: isPieChart ? undefined : {
        type: isBarChart ? 'category' : xNumeric ? 'value' : 'category',
        data: isBarChart ? barCategories : undefined,
        boundaryGap: isBarChart, // true for bar charts, false otherwise
        name: chartConfig.xAxis,
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: {
          color: '#4B5563',  // gray-600
          fontSize: 12
        },
        axisLabel: {
          color: '#6B7280',  // gray-500
          formatter: xNumeric ? (value) => value : null,
          margin: 8,
          rotate: xData.length > 10 ? 45 : 0,
          fontSize: 11
        },
        axisLine: {
          lineStyle: {
            color: '#D1D5DB',  // gray-300
          },
        },
        axisTick: {
          lineStyle: {
            color: '#E5E7EB',  // gray-200
          },
        },
        splitLine: {
          lineStyle: {
            color: '#F3F4F6',  // gray-100
            type: 'dashed'
          },
        },
      },
      yAxis: isPieChart ? undefined : [
        {
          type: 'value',
          name: chartConfig.yAxis,
          nameLocation: 'middle',
          nameGap: 50,
          nameTextStyle: {
            color: '#4B5563',  // gray-600
            fontSize: 12
          },
          axisLabel: {
            color: '#6B7280',  // gray-500
            margin: 8,
            fontSize: 11
          },
          axisLine: {
            lineStyle: {
              color: '#D1D5DB',  // gray-300
            },
          },
          axisTick: {
            lineStyle: {
              color: '#E5E7EB',  // gray-200
            },
          },
          splitLine: {
            lineStyle: {
              color: '#F3F4F6',  // gray-100
              type: 'dashed'
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
                color: '#7C3AED',  // violet-600
                fontSize: 12
              },
              axisLabel: {
                color: '#7C3AED',  // violet-600
                margin: 8,
                fontSize: 11
              },
              axisLine: {
                lineStyle: {
                  color: '#7C3AED',  // violet-600
                },
              },
              axisTick: {
                lineStyle: {
                  color: '#7C3AED',  // violet-600
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
        left: 'left',
        top: 5,
        textStyle: { 
          color: '#1F2937',  // gray-800
          fontSize: 14,
          fontWeight: 600
        },
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#1F2937',  // gray-800
        borderColor: '#374151',  // gray-700
        borderWidth: 1,
        textStyle: {
          color: '#F9FAFB',  // gray-50
          fontSize: 12
        },
      },
      grid: {
        left: '3%',
        right: '3%',
        bottom: '12%',
        top: '15%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: messageLabels,
        axisLabel: {
          color: '#6B7280',  // gray-500
          margin: 8,
          fontSize: 11
        },
        axisLine: {
          lineStyle: {
            color: '#D1D5DB',  // gray-300
          },
        },
        axisTick: {
          lineStyle: {
            color: '#E5E7EB',  // gray-200
          },
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          color: '#6B7280',  // gray-500
          margin: 8,
          fontSize: 11
        },
        axisLine: {
          lineStyle: {
            color: '#D1D5DB',  // gray-300
          },
        },
        axisTick: {
          lineStyle: {
            color: '#E5E7EB',  // gray-200
          },
        },
        splitLine: {
          lineStyle: {
            color: '#F3F4F6',  // gray-100
            type: 'dashed'
          },
        },
        scale: true,
      },
      toolbox: {
        feature: {
          saveAsImage: {
            icon: 'path://M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l2.007-7.454c.158-.474.457-.85.85-1.093A2.25 2.25 0 018 6h8v9zm-1-7a1 1 0 10-2 0 1 1 0 002 0z',
            title: 'Save',
            iconStyle: {
              color: '#6B7280',  // gray-500
              borderColor: '#E5E7EB',  // gray-200
              borderWidth: 1,
            },
            emphasis: {
              iconStyle: {
                color: '#4B5563',  // gray-600
              },
            },
          },
        },
        right: 10,
        top: 5,
        itemSize: 16,
        itemGap: 8,
      },
      series: [
        {
          type: 'line',
          data: messagesData,
          smooth: true,
          itemStyle: {
            color: '#2563EB',  // blue-600
          },
          lineStyle: {
            width: 2
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(37, 99, 235, 0.2)' },  // blue-600 with opacity
                { offset: 1, color: 'rgba(37, 99, 235, 0)' },
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
      <div className="min-h-screen">
        <Navbar />
        <div className="flex items-center justify-center mt-40"> {/* Adjusted top margin */}
          <div className="w-full max-w-3xl px-4 container rounded mt-20 p-10">
            <h2 className="text-2xl p-2 font-bold">Select a Simulation</h2>
            <p className="text-lg">
              Choose the simulation that you want to analyse.
            </p>

            {/* Simulation List as Cards */}
            <div className="mt-6">
              {simulationCatalog.map((sim) => (
                <div
                  key={sim.simulation_id}
                  onClick={() => setSimulationId(sim.simulation_id)}
                  className={`container p-3 my-2 flex justify-between items-center border rounded text-left cursor-pointer transition-colors duration-200 ${
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
            {error && <p className="danger p-2 rounded">{error}</p>}

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

  // Check if simulation has no runs (pending/processing)
  const isPending = simulationData && simulationData.runs.length === 0;
  
  // Determine simulation status
  const getSimulationStatus = () => {
    if (!simulationData) return 'loading';
    if (isPending) return 'pending';
    if (simulationData.runs.length > 0) return 'completed';
    return 'unknown';
  };
  
  const simulationStatus = getSimulationStatus();

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
            className="px-4 py-2 hover:shadow-button  rounded transition-colors flex items-center"
          >
            <TbArrowBackUp className="mr-2 h-5 w-5" />
            Catalog
          </Link>

          <button
            onClick={handleDownloadConfig}
            className="px-4 py-2 hover:shadow-button  rounded transition-colors flex items-center"
          >
            <IoMdSave className="mr-2 h-5 w-5" />
            Download&nbsp;JSON
          </button>
        </div>
      </div>
    
      {/* Simulation Info Panel */}
      <div className="">
        <div className="p-4 rounded mb-4 border bg-white/5">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-lg font-semibold">Simulation Details</h2>
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${
                simulationStatus === 'pending' ? 'bg-yellow-400 animate-pulse' : 
                simulationStatus === 'completed' ? 'bg-green-400' : 
                'bg-gray-400'
              }`} />
              <span className="text-sm capitalize">{simulationStatus}</span>
            </div>
          </div>
          
          <div className="space-y-1 text-sm">
            <p>
              <span className="font-medium text-gray-700 dark:text-gray-300">ID:</span>{' '}
              <span className="font-mono text-gray-900 dark:text-gray-100">{simulationId}</span>
            </p>
            
            {simulationStatus === 'pending' && (
              <div className="mt-3 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 rounded">
                <p className="text-yellow-800 dark:text-yellow-200">
                  <span className="font-medium">🔄 Processing:</span> Your simulation is currently running. 
                  Messages will appear in real-time below as agents interact.
                </p>
              </div>
            )}
            
            {simulationStatus === 'completed' && (
              <>
                <p>
                  <span className="font-medium text-gray-700 dark:text-gray-300">Agents:</span>{' '}
                  <span className="text-gray-900 dark:text-gray-100">
                    {Array.from(
                      new Set(
                        simulationData.runs
                          .flatMap((run) => run.messages?.map((msg) => msg.agent) || [])
                          .filter((agent) => agent !== 'InformationReturnAgent')
                      )
                    ).join(', ') || 'No agents found'}
                  </span>
                </p>
                <p>
                  <span className="font-medium text-gray-700 dark:text-gray-300">Total Runs:</span>{' '}
                  <span className="text-gray-900 dark:text-gray-100">{simulationData.runs.length}</span>
                </p>
                <p>
                  <span className="font-medium text-gray-700 dark:text-gray-300">Status:</span>{' '}
                  <span className="text-green-600 dark:text-green-400 font-medium">✓ Completed</span>
                </p>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Live Chat Stream */}
      <div className="mb-6">
        <ChatStream 
          simulationId={simulationId} 
          isSimulationComplete={!isPending && simulationData?.runs?.length > 0}
          totalRuns={simulationData?.runs?.length || 0}
        />
      </div>

      {/* Messages per Run Chart */}
      {!isPending && (
        <div className="mb-4 max-w-3xl">
          <ReactECharts option={generateMessageChartOption()} className="h-48 w-full" />
        </div>
      )}
      
      {/* Mode Selector */}
      <div className="mb-6 flex space-x-4">
        <button
          onClick={toggleAdvancedMode}
          className={`px-4 py-2 rounded-md font-medium transition-all ${
            !advancedMode
              ? 'bg-blue-600 text-white hover:bg-blue-700'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600'
          }`}
        >
          Basic Mode
        </button>
        <button
          onClick={toggleAdvancedMode}
          className={`px-4 py-2 rounded-md font-medium transition-all ${
            advancedMode
              ? 'bg-blue-600 text-white hover:bg-blue-700'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600'
          }`}
        >
          Advanced Mode
        </button>
      </div>

      {/* Chart Configuration Panel */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Chart Configuration</h2>
          <button
            onClick={() => toggleSection('chartConfig')}
            className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-md transition-colors"
          >
            {collapsibleSections.chartConfig ? (
              <IoEyeOffOutline size={20} className="text-gray-700 dark:text-gray-300" />
            ) : (
              <IoEyeOutline size={20} className="text-gray-700 dark:text-gray-300" />
            )}
          </button>
        </div>

        {collapsibleSections.chartConfig && (
          <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Chart Type Selection */}
              <div>
                <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">Chart Type</label>
                <select
                  value={chartConfig.type}
                  onChange={(e) => setChartConfig({ ...chartConfig, type: e.target.value })}
                  className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500"
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
                <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">X-Axis Variable</label>
                <select
                  value={chartConfig.xAxis}
                  onChange={(e) => setChartConfig({ ...chartConfig, xAxis: e.target.value })}
                  className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500"
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
                <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">Y-Axis Variable</label>
                <select
                  value={chartConfig.yAxis}
                  onChange={(e) => setChartConfig({ ...chartConfig, yAxis: e.target.value })}
                  className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500"
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
                <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">Visualization Options</label>
                <div className="flex flex-col space-y-2">
                  <label className="inline-flex items-center text-sm text-gray-700 dark:text-gray-300">
                    <input
                      type="checkbox"
                      checked={chartConfig.showTrendline}
                      onChange={(e) =>
                        setChartConfig({ ...chartConfig, showTrendline: e.target.checked })
                      }
                      className="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    Show Trendline
                  </label>
                  <label className="inline-flex items-center text-sm text-gray-700 dark:text-gray-300">
                    <input
                      type="checkbox"
                      checked={chartConfig.showLegend}
                      onChange={(e) =>
                        setChartConfig({ ...chartConfig, showLegend: e.target.checked })
                      }
                      className="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    Show Legend
                  </label>
                  <label className="inline-flex items-center text-sm text-gray-700 dark:text-gray-300">
                    <input
                      type="checkbox"
                      checked={chartConfig.showDataLabels}
                      onChange={(e) =>
                        setChartConfig({ ...chartConfig, showDataLabels: e.target.checked })
                      }
                      className="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
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

      {/* All Variables Summary */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">All Variables Summary</h2>
          <button
            onClick={() => toggleSection('variables')}
            className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-md transition-colors"
          >
            {collapsibleSections.variables ? (
              <IoEyeOffOutline size={20} className="text-gray-700 dark:text-gray-300" />
            ) : (
              <IoEyeOutline size={20} className="text-gray-700 dark:text-gray-300" />
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
                  className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-4 rounded-lg shadow-sm hover:shadow-md cursor-pointer transition-all duration-200 hover:border-blue-400 dark:hover:border-blue-500"
                  onClick={() => setChartConfig({ ...chartConfig, yAxis: variable })}
                >
                  <h3 className="font-medium text-lg mb-2 text-gray-900 dark:text-gray-100">{variable}</h3>
                  {numeric && stats ? (
                    <div className="space-y-1">
                      <p className="text-sm text-gray-700 dark:text-gray-300">
                        <span className="font-medium">Mean:</span> {stats.mean}
                      </p>
                      <p className="text-sm text-gray-700 dark:text-gray-300">
                        <span className="font-medium">Range:</span> {stats.min} - {stats.max}
                      </p>
                      <p className="text-sm text-gray-700 dark:text-gray-300">
                        <span className="font-medium">StdDev:</span> {stats.stdDev}
                      </p>
                    </div>
                  ) : (
                    <p className="text-sm text-gray-700 dark:text-gray-300">
                      {values.slice(0, 3).map((v, i) => {
                        if (typeof v === 'object' && v !== null) {
                          if (variable === 'utility' && !Array.isArray(v)) {
                            // Format utility objects
                            const entries = Object.entries(v)
                              .filter(([k, val]) => typeof val === 'number')
                              .map(([k, val]) => `${k}: ${val.toFixed(3)}`);
                            return entries.join(', ');
                          }
                          return JSON.stringify(v);
                        }
                        return String(v);
                      }).join(' | ')}
                      {values.length > 3 ? ' ...' : ''}
                    </p>
                  )}
                  <div className="mt-3 text-xs text-blue-600 dark:text-blue-400 font-medium">Click to set as Y-axis</div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Main Visualization */}
      <div className="mb-4 max-w-4xl">
        {chartConfig.xAxis && chartConfig.yAxis ? (
          <div className="bg-gray-900/5 dark:bg-gray-800 p-3 rounded-lg border border-gray-300 dark:border-gray-700 shadow-sm">
            <ReactECharts
              option={generateChartOptions()}
              style={{ height: '320px', width: '100%' }}
              className="bg-transparent"
              opts={{ renderer: 'canvas' }}
            />
          </div>
        ) : (
          <div className="bg-gray-100 dark:bg-gray-800/50 border-2 border-dashed border-gray-400 dark:border-gray-600 rounded-lg p-8 text-center">
            <p className="text-base font-medium text-gray-800 dark:text-gray-300 mb-1">Please select variables for X and Y axes to visualize data.</p>
            <p className="text-sm text-gray-600 dark:text-gray-500">
              Use the Chart Configuration panel above to customize your visualization.
            </p>
          </div>
        )}
      </div>

      {/* Statistics Panel */}
      {chartConfig.yAxis && isDataNumeric(getDataForVariable(chartConfig.yAxis)) && (
        <div className="mb-6">
          <div className="flex justify-between items-center mb-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/10 dark:to-purple-900/10 rounded-lg border border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Statistics for {chartConfig.yAxis}
            </h2>
            <button
              onClick={() => toggleSection('statistics')}
              className="p-2 hover:bg-white/50 dark:hover:bg-gray-800/50 rounded-full transition-all duration-200"
            >
              {collapsibleSections.statistics ? (
                <IoEyeOffOutline size={20} className="text-gray-600 dark:text-gray-400" />
              ) : (
                <IoEyeOutline size={20} className="text-gray-600 dark:text-gray-400" />
              )}
            </button>
          </div>

          {collapsibleSections.statistics && (
            <div className="bg-white dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700 p-6 shadow-sm">
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                {Object.entries(computeStats(chartConfig.yAxis) || {}).map(([stat, value]) => {
                  // Define icons and colors for each stat
                  const statConfig = {
                    count: { icon: '📊', color: 'from-blue-400 to-blue-600', bgColor: 'bg-blue-50 dark:bg-blue-900/20' },
                    mean: { icon: '📈', color: 'from-green-400 to-green-600', bgColor: 'bg-green-50 dark:bg-green-900/20' },
                    median: { icon: '🎯', color: 'from-purple-400 to-purple-600', bgColor: 'bg-purple-50 dark:bg-purple-900/20' },
                    stdDev: { icon: '📉', color: 'from-orange-400 to-orange-600', bgColor: 'bg-orange-50 dark:bg-orange-900/20' },
                    min: { icon: '⬇️', color: 'from-red-400 to-red-600', bgColor: 'bg-red-50 dark:bg-red-900/20' },
                    max: { icon: '⬆️', color: 'from-indigo-400 to-indigo-600', bgColor: 'bg-indigo-50 dark:bg-indigo-900/20' }
                  };
                  
                  const config = statConfig[stat] || { icon: '📊', color: 'from-gray-400 to-gray-600', bgColor: 'bg-gray-50 dark:bg-gray-900/20' };
                  
                  return (
                    <div
                      key={stat}
                      className={`${config.bgColor} rounded-lg p-4 border border-gray-200 dark:border-gray-700 hover:shadow-md transition-all duration-200 transform hover:scale-105`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-2xl">{config.icon}</span>
                        <h3 className="text-xs font-medium text-gray-600 dark:text-gray-400 uppercase tracking-wide">
                          {stat === 'stdDev' ? 'Std Dev' : stat.charAt(0).toUpperCase() + stat.slice(1)}
                        </h3>
                      </div>
                      <p className={`text-2xl font-bold bg-gradient-to-r ${config.color} bg-clip-text text-transparent`}>
                        {value}
                      </p>
                    </div>
                  );
                })}
              </div>

              {secondaryYAxis.enabled && secondaryYAxis.variable && (
                <div className="mt-8">
                  <h3 className="text-lg font-bold mb-4 bg-gradient-to-r from-teal-600 to-cyan-600 bg-clip-text text-transparent">
                    Statistics for {secondaryYAxis.variable} (Secondary Y-Axis)
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                    {Object.entries(computeStats(secondaryYAxis.variable) || {}).map(
                      ([stat, value]) => {
                        const statConfig = {
                          count: { icon: '📊', color: 'from-teal-400 to-teal-600', bgColor: 'bg-teal-50 dark:bg-teal-900/20' },
                          mean: { icon: '📈', color: 'from-cyan-400 to-cyan-600', bgColor: 'bg-cyan-50 dark:bg-cyan-900/20' },
                          median: { icon: '🎯', color: 'from-sky-400 to-sky-600', bgColor: 'bg-sky-50 dark:bg-sky-900/20' },
                          stdDev: { icon: '📉', color: 'from-lime-400 to-lime-600', bgColor: 'bg-lime-50 dark:bg-lime-900/20' },
                          min: { icon: '⬇️', color: 'from-rose-400 to-rose-600', bgColor: 'bg-rose-50 dark:bg-rose-900/20' },
                          max: { icon: '⬆️', color: 'from-violet-400 to-violet-600', bgColor: 'bg-violet-50 dark:bg-violet-900/20' }
                        };
                        
                        const config = statConfig[stat] || { icon: '📊', color: 'from-gray-400 to-gray-600', bgColor: 'bg-gray-50 dark:bg-gray-900/20' };
                        
                        return (
                          <div
                            key={stat}
                            className={`${config.bgColor} rounded-lg p-4 border border-gray-200 dark:border-gray-700 hover:shadow-md transition-all duration-200 transform hover:scale-105`}
                          >
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-2xl">{config.icon}</span>
                              <h3 className="text-xs font-medium text-gray-600 dark:text-gray-400 uppercase tracking-wide">
                                {stat === 'stdDev' ? 'Std Dev' : stat.charAt(0).toUpperCase() + stat.slice(1)}
                              </h3>
                            </div>
                            <p className={`text-2xl font-bold bg-gradient-to-r ${config.color} bg-clip-text text-transparent`}>
                              {value}
                            </p>
                          </div>
                        );
                      }
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Data Table */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Data Table</h2>
          <button
            onClick={() => toggleSection('dataTable')}
            className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-md transition-colors"
          >
            {collapsibleSections.dataTable ? (
              <IoEyeOffOutline size={20} className="text-gray-700 dark:text-gray-300" />
            ) : (
              <IoEyeOutline size={20} className="text-gray-700 dark:text-gray-300" />
            )}
          </button>
        </div>

        {collapsibleSections.dataTable && (
          <div className="overflow-x-auto bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-900">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 dark:text-gray-300 uppercase tracking-wider">
                    Run
                  </th>
                  {availableVariables.map((variable) => (
                    <th
                      key={`header-${variable}`}
                      className="px-6 py-3 text-left text-xs font-medium text-gray-700 dark:text-gray-300 uppercase tracking-wider"
                    >
                      {variable}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {simulationData.runs.map((run, index) => (
                  <tr key={`run-${index}`} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
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
                        
                        // Handle object values for display
                        if (typeof value === 'object' && value !== null) {
                          if (variable === 'utility' && !Array.isArray(value)) {
                            // Format utility objects nicely
                            const entries = Object.entries(value)
                              .filter(([k, v]) => typeof v === 'number')
                              .map(([k, v]) => `${k}: ${typeof v === 'number' ? v.toFixed(3) : v}`);
                            value = entries.join(', ');
                          } else if (Array.isArray(value)) {
                            value = `[${value.join(', ')}]`;
                          } else {
                            // Try to extract a meaningful value or show JSON
                            if ('value' in value) value = value.value;
                            else if ('amount' in value) value = value.amount;
                            else if ('score' in value) value = value.score;
                            else if ('result' in value) value = value.result;
                            else value = JSON.stringify(value);
                          }
                        }
                        
                        // Format numbers nicely
                        if (typeof value === 'number') {
                          value = value % 1 === 0 ? value : value.toFixed(3);
                        }
                      }

                      return (
                        <td
                          key={`cell-${index}-${variable}`}
                          className="px-6 py-4 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300"
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
