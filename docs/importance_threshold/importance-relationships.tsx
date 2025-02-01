import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

const ImportanceRelationships = () => {
  // Generate data for importance vs magnitude at different std ratios
  const generateMagnitudeData = () => {
    const data = [];
    for (let magnitude = 0.0001; magnitude <= 0.001; magnitude += 0.00002) {
      const point = {
        magnitude,
        perfect: magnitude * (1 / (1 + 0.1)), // Almost perfect consistency
        good: magnitude * (1 / (1 + 0.5)),    // Good consistency
        typical: magnitude * (1 / (1 + 1.0)),  // Typical consistency
        poor: magnitude * (1 / (1 + 2.0))      // Poor consistency
      };
      data.push(point);
    }
    return data;
  };

  // Generate data for importance vs std ratio at different magnitudes
  const generateStdRatioData = () => {
    const data = [];
    for (let stdRatio = 0; stdRatio <= 3; stdRatio += 0.1) {
      const point = {
        stdRatio,
        strong: 0.001 * (1 / (1 + stdRatio)),    // Strong effect
        moderate: 0.0005 * (1 / (1 + stdRatio)),  // Moderate effect
        weak: 0.00025 * (1 / (1 + stdRatio))      // Weak effect
      };
      data.push(point);
    }
    return data;
  };

  const magnitudeData = generateMagnitudeData();
  const stdRatioData = generateStdRatioData();

  return (
    <div className="w-full p-6 space-y-8">
      {/* First plot: Importance vs Magnitude */}
      <div>
        <h3 className="text-lg font-semibold mb-2">Importance vs Effect Magnitude</h3>
        <p className="text-sm text-gray-600 mb-4">
          How importance grows with effect magnitude at different consistency levels
        </p>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={magnitudeData} margin={{ top: 20, right: 30, left: 60, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="magnitude" 
                tickFormatter={(value) => value.toFixed(4)}
                label={{ value: "Effect Magnitude", position: "bottom", offset: 10 }}
              />
              <YAxis 
                label={{ value: "Importance", angle: -90, position: "insideLeft", offset: -40 }}
                tickFormatter={(value) => value.toFixed(4)}
              />
              <Tooltip formatter={(value) => value.toFixed(6)} />
              <ReferenceLine y={0.00025} stroke="red" strokeDasharray="3 3" label="Threshold" />
              <Line type="monotone" dataKey="perfect" stroke="#2196F3" name="Perfect Consistency (σ/μ = 0.1)" dot={false} />
              <Line type="monotone" dataKey="good" stroke="#4CAF50" name="Good Consistency (σ/μ = 0.5)" dot={false} />
              <Line type="monotone" dataKey="typical" stroke="#FFC107" name="Typical Consistency (σ/μ = 1.0)" dot={false} />
              <Line type="monotone" dataKey="poor" stroke="#FF5252" name="Poor Consistency (σ/μ = 2.0)" dot={false} />
              <Legend />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Second plot: Importance vs Std Ratio */}
      <div>
        <h3 className="text-lg font-semibold mb-2">Importance vs Consistency</h3>
        <p className="text-sm text-gray-600 mb-4">
          How importance decays with increasing variability (σ/μ ratio) for different effect magnitudes
        </p>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={stdRatioData} margin={{ top: 20, right: 30, left: 60, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="stdRatio"
                label={{ value: "Standard Deviation Ratio (σ/μ)", position: "bottom", offset: 10 }}
              />
              <YAxis 
                label={{ value: "Importance", angle: -90, position: "insideLeft", offset: -40 }}
                tickFormatter={(value) => value.toFixed(4)}
              />
              <Tooltip formatter={(value) => value.toFixed(6)} />
              <ReferenceLine y={0.00025} stroke="red" strokeDasharray="3 3" label="Threshold" />
              <Line type="monotone" dataKey="strong" stroke="#2196F3" name="Strong Effect (μ = 0.001)" dot={false} />
              <Line type="monotone" dataKey="moderate" stroke="#4CAF50" name="Moderate Effect (μ = 0.0005)" dot={false} />
              <Line type="monotone" dataKey="weak" stroke="#FFC107" name="Weak Effect (μ = 0.00025)" dot={false} />
              <Legend />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="text-sm space-y-2">
        <p><strong>Implications of 0.00025 Threshold:</strong></p>
        <ul className="list-disc pl-5 space-y-1">
          <li>Features need larger effect magnitudes to compensate for higher variability</li>
          <li>At typical consistency levels (σ/μ = 1.0), need magnitude ≈ 0.0005 to exceed threshold</li>
          <li>Even strong effects become unimportant if too variable (σ/μ > 2)</li>
          <li>The threshold naturally separates reliable effects from noise</li>
        </ul>
      </div>
    </div>
  );
};

export default ImportanceRelationships;