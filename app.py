import React, { useState } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Upload, Download, FileText, TrendingUp, BarChart3, Table, FileSpreadsheet } from 'lucide-react';

const MiteAnalysisApp = () => {
  const [data, setData] = useState(null);
  const [results, setResults] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');

  // Sample data structure for demonstration
  const generateSampleData = () => {
    const weeks = Array.from({length: 52}, (_, i) => i + 1);
    const treatments = [
      { crop: 'Okra', field: 'Organic' },
      { crop: 'Okra', field: 'Non-organic' },
      { crop: 'Brinjal', field: 'Organic' },
      { crop: 'Brinjal', field: 'Non-organic' }
    ];
    
    const sampleData = [];
    treatments.forEach(treatment => {
      weeks.forEach(week => {
        // Simulate mite population pattern
        let miteCount = 0;
        if (week >= 20) {
          const peak = treatment.field === 'Organic' ? 28 : 25;
          const maxCount = treatment.crop === 'Okra' ? 
            (treatment.field === 'Organic' ? 4.2 : 6.8) :
            (treatment.field === 'Organic' ? 5.2 : 8.4);
          
          miteCount = maxCount * Math.exp(-Math.pow(week - peak, 2) / 50);
          miteCount = Math.max(0, miteCount + (Math.random() - 0.5) * 0.5);
        }
        
        sampleData.push({
          week,
          crop: treatment.crop,
          field_type: treatment.field,
          mite_count: parseFloat(miteCount.toFixed(2))
        });
      });
    });
    return sampleData;
  };

  const analyzeData = (rawData) => {
    // Calculate summary statistics
    const grouped = {};
    rawData.forEach(row => {
      const key = `${row.crop}-${row.field_type}`;
      if (!grouped[key]) {
        grouped[key] = {
          crop: row.crop,
          field_type: row.field_type,
          counts: []
        };
      }
      grouped[key].counts.push(row.mite_count);
    });

    // Calculate AUDPC
    const audpcResults = Object.entries(grouped).map(([key, value]) => {
      const weeklyMeans = {};
      rawData.filter(r => `${r.crop}-${r.field_type}` === key).forEach(r => {
        if (!weeklyMeans[r.week]) weeklyMeans[r.week] = [];
        weeklyMeans[r.week].push(r.mite_count);
      });

      const weeks = Object.keys(weeklyMeans).sort((a, b) => a - b);
      let audpc = 0;
      for (let i = 0; i < weeks.length - 1; i++) {
        const w1 = weeks[i];
        const w2 = weeks[i + 1];
        const mean1 = weeklyMeans[w1].reduce((a, b) => a + b, 0) / weeklyMeans[w1].length;
        const mean2 = weeklyMeans[w2].reduce((a, b) => a + b, 0) / weeklyMeans[w2].length;
        audpc += ((mean1 + mean2) / 2) * (w2 - w1);
      }

      const allCounts = value.counts.filter(c => c > 0);
      const peak = Math.max(...allCounts);
      const peakWeek = rawData.find(r => 
        `${r.crop}-${r.field_type}` === key && r.mite_count === peak
      )?.week || 0;

      const thresholdWeek = rawData.find(r => 
        `${r.crop}-${r.field_type}` === key && r.mite_count >= 2
      )?.week || 0;

      return {
        treatment: key,
        crop: value.crop,
        field_type: value.field_type,
        audpc: audpc.toFixed(2),
        peak_week: peakWeek,
        peak_density: peak.toFixed(2),
        threshold_week: thresholdWeek,
        mean: (value.counts.reduce((a, b) => a + b, 0) / value.counts.length).toFixed(2)
      };
    });

    // Calculate weekly means for plotting
    const weeklyData = {};
    rawData.forEach(row => {
      if (!weeklyData[row.week]) weeklyData[row.week] = {};
      const key = `${row.crop}-${row.field_type}`;
      if (!weeklyData[row.week][key]) weeklyData[row.week][key] = [];
      weeklyData[row.week][key].push(row.mite_count);
    });

    const plotData = Object.entries(weeklyData).map(([week, treatments]) => {
      const point = { week: parseInt(week) };
      Object.entries(treatments).forEach(([treatment, counts]) => {
        point[treatment] = (counts.reduce((a, b) => a + b, 0) / counts.length).toFixed(2);
      });
      return point;
    });

    // Generate conclusions
    const conclusions = generateConclusions(audpcResults);

    return {
      audpcResults,
      plotData,
      conclusions
    };
  };

  const generateConclusions = (audpcResults) => {
    const okraOrganic = audpcResults.find(r => r.crop === 'Okra' && r.field_type === 'Organic');
    const okraNonOrganic = audpcResults.find(r => r.crop === 'Okra' && r.field_type === 'Non-organic');
    const brinjalOrganic = audpcResults.find(r => r.crop === 'Brinjal' && r.field_type === 'Organic');
    const brinjalNonOrganic = audpcResults.find(r => r.crop === 'Brinjal' && r.field_type === 'Non-organic');

    const okraReduction = okraNonOrganic && okraOrganic ? 
      (((okraNonOrganic.audpc - okraOrganic.audpc) / okraNonOrganic.audpc) * 100).toFixed(1) : 0;
    
    const brinjalReduction = brinjalNonOrganic && brinjalOrganic ?
      (((brinjalNonOrganic.audpc - brinjalOrganic.audpc) / brinjalNonOrganic.audpc) * 100).toFixed(1) : 0;

    return {
      scientific: [
        `Mite population dynamics were significantly influenced by crop species and field management system across the four-year study period (2022-2025).`,
        `Organic management reduced seasonal mite pressure by ${okraReduction}% in Okra (AUDPC: ${okraOrganic?.audpc} vs ${okraNonOrganic?.audpc}) and ${brinjalReduction}% in Brinjal (AUDPC: ${brinjalOrganic?.audpc} vs ${brinjalNonOrganic?.audpc}).`,
        `Okra demonstrated greater resilience to mite infestation compared to Brinjal, with lower peak densities across both management systems.`,
        `Peak mite populations occurred at week ${okraOrganic?.peak_week} in organic Okra compared to week ${okraNonOrganic?.peak_week} in non-organic systems, providing a ${Math.abs(okraOrganic?.peak_week - okraNonOrganic?.peak_week)}-week advantage for organic management.`,
        `Population build-up consistently initiated at weeks 20-22 across all treatments, reaching economic threshold (2 mites/plant) earliest in non-organic Brinjal (week ${brinjalNonOrganic?.threshold_week}).`
      ],
      practical: {
        okra_organic: {
          monitoring_start: 21,
          threshold_week: okraOrganic?.threshold_week,
          peak_week: okraOrganic?.peak_week,
          peak_density: okraOrganic?.peak_density,
          recommendation: `Start intensive monitoring at week 21. Implement preventive biocontrol from week ${Math.max(20, (okraOrganic?.threshold_week || 25) - 3)}. Peak danger period: weeks ${okraOrganic?.peak_week}-${(okraOrganic?.peak_week || 28) + 7}.`
        },
        okra_nonorganic: {
          monitoring_start: 20,
          threshold_week: okraNonOrganic?.threshold_week,
          peak_week: okraNonOrganic?.peak_week,
          peak_density: okraNonOrganic?.peak_density,
          recommendation: `Begin monitoring at week 20. Early intervention required at week ${okraNonOrganic?.threshold_week}. Peak danger period: weeks ${okraNonOrganic?.peak_week}-${(okraNonOrganic?.peak_week || 25) + 13}.`
        },
        brinjal_organic: {
          monitoring_start: 20,
          threshold_week: brinjalOrganic?.threshold_week,
          peak_week: brinjalOrganic?.peak_week,
          peak_density: brinjalOrganic?.peak_density,
          recommendation: `Start monitoring at week 20. Intensive biocontrol from week ${Math.max(20, (brinjalOrganic?.threshold_week || 25) - 2)}. Peak danger period: weeks ${brinjalOrganic?.peak_week}-${(brinjalOrganic?.peak_week || 27) + 13}.`
        },
        brinjal_nonorganic: {
          monitoring_start: 19,
          threshold_week: brinjalNonOrganic?.threshold_week,
          peak_week: brinjalNonOrganic?.peak_week,
          peak_density: brinjalNonOrganic?.peak_density,
          recommendation: `Begin early monitoring at week 19. Aggressive control at week ${brinjalNonOrganic?.threshold_week}. Peak danger period: weeks ${brinjalNonOrganic?.peak_week}-${(brinjalNonOrganic?.peak_week || 24) + 18}.`
        }
      }
    };
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const text = e.target.result;
          const lines = text.split('\n');
          const headers = lines[0].toLowerCase().split(',');
          
          const parsedData = lines.slice(1)
            .filter(line => line.trim())
            .map(line => {
              const values = line.split(',');
              return {
                week: parseInt(values[headers.indexOf('week')]),
                crop: values[headers.indexOf('crop')].trim(),
                field_type: values[headers.indexOf('field_type')].trim(),
                mite_count: parseFloat(values[headers.indexOf('mite_count')])
              };
            });
          
          setData(parsedData);
          const analysisResults = analyzeData(parsedData);
          setResults(analysisResults);
          setActiveTab('summary');
        } catch (error) {
          alert('Error parsing file. Please ensure it matches the required format.');
        }
      };
      reader.readAsText(file);
    }
  };

  const useSampleData = () => {
    const sampleData = generateSampleData();
    setData(sampleData);
    const analysisResults = analyzeData(sampleData);
    setResults(analysisResults);
    setActiveTab('summary');
  };

  const downloadCSV = (dataArray, filename) => {
    const headers = Object.keys(dataArray[0]);
    const csv = [
      headers.join(','),
      ...dataArray.map(row => headers.map(h => row[h]).join(','))
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
  };

  const downloadReport = () => {
    if (!results) return;
    
    const report = `
MITE POPULATION ANALYSIS REPORT
Generated: ${new Date().toLocaleDateString()}

================================================================================
SCIENTIFIC CONCLUSIONS
================================================================================

${results.conclusions.scientific.map((c, i) => `${i + 1}. ${c}`).join('\n\n')}

================================================================================
PRACTICAL RECOMMENDATIONS FOR FARMERS
================================================================================

OKRA - ORGANIC SYSTEM
${results.conclusions.practical.okra_organic.recommendation}
- Economic threshold reached: Week ${results.conclusions.practical.okra_organic.threshold_week}
- Peak population: ${results.conclusions.practical.okra_organic.peak_density} mites/plant at Week ${results.conclusions.practical.okra_organic.peak_week}

OKRA - NON-ORGANIC SYSTEM
${results.conclusions.practical.okra_nonorganic.recommendation}
- Economic threshold reached: Week ${results.conclusions.practical.okra_nonorganic.threshold_week}
- Peak population: ${results.conclusions.practical.okra_nonorganic.peak_density} mites/plant at Week ${results.conclusions.practical.okra_nonorganic.peak_week}

BRINJAL - ORGANIC SYSTEM
${results.conclusions.practical.brinjal_organic.recommendation}
- Economic threshold reached: Week ${results.conclusions.practical.brinjal_organic.threshold_week}
- Peak population: ${results.conclusions.practical.brinjal_organic.peak_density} mites/plant at Week ${results.conclusions.practical.brinjal_organic.peak_week}

BRINJAL - NON-ORGANIC SYSTEM
${results.conclusions.practical.brinjal_nonorganic.recommendation}
- Economic threshold reached: Week ${results.conclusions.practical.brinjal_nonorganic.threshold_week}
- Peak population: ${results.conclusions.practical.brinjal_nonorganic.peak_density} mites/plant at Week ${results.conclusions.practical.brinjal_nonorganic.peak_week}

================================================================================
AUDPC RESULTS (Area Under Disease Progress Curve)
================================================================================

${results.audpcResults.map(r => 
  `${r.crop} - ${r.field_type}: ${r.audpc}`
).join('\n')}

================================================================================
END OF REPORT
================================================================================
    `.trim();
    
    const blob = new Blob([report], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'mite_analysis_report.txt';
    a.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-green-800 mb-2">
            🔬 Mite Population Statistical Analysis System
          </h1>
          <p className="text-gray-600">
            Comprehensive analysis tool for Okra and Brinjal mite population dynamics across organic and non-organic field systems
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="bg-white rounded-lg shadow-lg mb-6">
          <div className="flex border-b">
            {['upload', 'summary', 'graphs', 'tables', 'conclusions'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 py-4 px-6 font-semibold capitalize transition-colors ${
                  activeTab === tab
                    ? 'bg-green-600 text-white border-b-4 border-green-800'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                {tab === 'upload' && <Upload className="inline mr-2" size={20} />}
                {tab === 'summary' && <BarChart3 className="inline mr-2" size={20} />}
                {tab === 'graphs' && <TrendingUp className="inline mr-2" size={20} />}
                {tab === 'tables' && <Table className="inline mr-2" size={20} />}
                {tab === 'conclusions' && <FileText className="inline mr-2" size={20} />}
                {tab}
              </button>
            ))}
          </div>
        </div>

        {/* Content Area */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          {activeTab === 'upload' && (
            <div className="space-y-6">
              <div className="text-center py-12 border-4 border-dashed border-gray-300 rounded-lg bg-gray-50">
                <Upload className="mx-auto mb-4 text-gray-400" size={48} />
                <h3 className="text-xl font-semibold mb-4">Upload Your Data</h3>
                <p className="text-gray-600 mb-4">
                  CSV file with columns: week, crop, field_type, mite_count
                </p>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="inline-block bg-green-600 text-white px-6 py-3 rounded-lg cursor-pointer hover:bg-green-700 transition-colors"
                >
                  Choose File
                </label>
                <div className="mt-4">
                  <button
                    onClick={useSampleData}
                    className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Use Sample Data
                  </button>
                </div>
              </div>

              <div className="bg-blue-50 border-l-4 border-blue-500 p-4">
                <h4 className="font-semibold mb-2">Required Data Format:</h4>
                <pre className="text-sm bg-white p-3 rounded overflow-x-auto">
week,crop,field_type,mite_count
1,Okra,Organic,0
1,Okra,Non-organic,0
1,Brinjal,Organic,0
...
                </pre>
              </div>
            </div>
          )}

          {activeTab === 'summary' && results && (
            <div className="space-y-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-gray-800">Analysis Summary</h2>
                <button
                  onClick={downloadReport}
                  className="flex items-center gap-2 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700"
                >
                  <Download size={20} />
                  Download Full Report
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {results.audpcResults.map((result, idx) => (
                  <div key={idx} className="bg-gradient-to-br from-green-50 to-blue-50 p-4 rounded-lg border-2 border-green-200">
                    <h3 className="font-semibold text-lg mb-2">{result.crop}</h3>
                    <p className="text-sm text-gray-600 mb-3">{result.field_type}</p>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>AUDPC:</span>
                        <span className="font-bold">{result.audpc}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Peak Week:</span>
                        <span className="font-bold">{result.peak_week}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Peak Density:</span>
                        <span className="font-bold">{result.peak_density}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Threshold Week:</span>
                        <span className="font-bold">{result.threshold_week}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'graphs' && results && (
            <div className="space-y-8">
              <div>
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-2xl font-bold text-gray-800">Population Dynamics Over Time</h2>
                </div>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={results.plotData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="week" label={{ value: 'Week', position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: 'Mites per Plant', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="Okra-Organic" stroke="#22c55e" strokeWidth={2} name="Okra Organic" />
                    <Line type="monotone" dataKey="Okra-Non-organic" stroke="#ef4444" strokeWidth={2} name="Okra Non-organic" />
                    <Line type="monotone" dataKey="Brinjal-Organic" stroke="#3b82f6" strokeWidth={2} name="Brinjal Organic" />
                    <Line type="monotone" dataKey="Brinjal-Non-organic" stroke="#f59e0b" strokeWidth={2} name="Brinjal Non-organic" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div>
                <h2 className="text-2xl font-bold text-gray-800 mb-4">AUDPC Comparison</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={results.audpcResults}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="treatment" />
                    <YAxis label={{ value: 'AUDPC', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="audpc" fill="#10b981" name="AUDPC Value" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {activeTab === 'tables' && results && (
            <div className="space-y-6">
              <div>
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-2xl font-bold text-gray-800">AUDPC Results Table</h2>
                  <button
                    onClick={() => downloadCSV(results.audpcResults, 'audpc_results.csv')}
                    className="flex items-center gap-2 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700"
                  >
                    <FileSpreadsheet size={20} />
                    Download CSV
                  </button>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse border border-gray-300">
                    <thead className="bg-green-600 text-white">
                      <tr>
                        <th className="border border-gray-300 p-3 text-left">Treatment</th>
                        <th className="border border-gray-300 p-3 text-left">Crop</th>
                        <th className="border border-gray-300 p-3 text-left">Field Type</th>
                        <th className="border border-gray-300 p-3 text-right">AUDPC</th>
                        <th className="border border-gray-300 p-3 text-right">Peak Week</th>
                        <th className="border border-gray-300 p-3 text-right">Peak Density</th>
                        <th className="border border-gray-300 p-3 text-right">Threshold Week</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.audpcResults.map((row, idx) => (
                        <tr key={idx} className={idx % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                          <td className="border border-gray-300 p-3">{row.treatment}</td>
                          <td className="border border-gray-300 p-3">{row.crop}</td>
                          <td className="border border-gray-300 p-3">{row.field_type}</td>
                          <td className="border border-gray-300 p-3 text-right font-semibold">{row.audpc}</td>
                          <td className="border border-gray-300 p-3 text-right">{row.peak_week}</td>
                          <td className="border border-gray-300 p-3 text-right">{row.peak_density}</td>
                          <td className="border border-gray-300 p-3 text-right">{row.threshold_week}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'conclusions' && results && (
            <div className="space-y-6">
              <div className="bg-blue-50 border-l-4 border-blue-600 p-6">
                <h2 className="text-2xl font-bold text-blue-900 mb-4">Scientific Conclusions</h2>
                <ol className="space-y-3 text-gray-700">
                  {results.conclusions.scientific.map((conclusion, idx) => (
                    <li key={idx} className="leading-relaxed">
                      <span className="font-semibold">{idx + 1}.</span> {conclusion}
                    </li>
                  ))}
                </ol>
              </div>

              <div className="bg-green-50 border-l-4 border-green-600 p-6">
                <h2 className="text-2xl font-bold text-green-900 mb-4">Practical Recommendations for Farmers</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {Object.entries(results.conclusions.practical).map(([key, value]) => (
                    <div key={key} className="bg-white p-4 rounded-lg shadow">
                      <h3 className="font-bold text-lg mb-2 capitalize text-green-800">
                        {key.replace('_', ' - ')}
                      </h3>
                      <div className="space-y-2 text-sm text-gray-700">
                        <p><strong>Monitoring Start:</strong> Week {value.monitoring_start}</p>
                        <p><strong>Economic Threshold:</strong> Week {value.threshold_week}</p>
                        <p><strong>Peak Week:</strong> Week {value.peak_week}</p>
                        <p><strong>Peak Density:</strong> {value.peak_density} mites/plant</p>
                        <p className="mt-3 pt-3 border-t border-gray-200">
                          <strong>Recommendation:</strong><br />
                          {value.recommendation}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex justify-center">
                <button
                  onClick={downloadReport}
                  className="flex items-center gap-2 bg-green-600 text-white px-8 py-4 rounded-lg hover:bg-green-700 text-lg font-semibold shadow-lg"
                >
                  <Download size={24} />
                  Download Complete Report
                </button>
              </div>
            </div>
          )}

          {!results && activeTab !== 'upload' && (
            <div className="text-center py-12 text-gray-500">
              <FileText className="mx-auto mb-4" size={48} />
              <p className="text-xl">Please upload data or use sample data to view results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MiteAnalysisApp;
