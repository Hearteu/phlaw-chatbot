"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart3, FileText, TrendingUp, Users } from "lucide-react";
import { useEffect, useState } from "react";

interface MetricsData {
  period_days: number;
  total_ratings: number;
  overall_metrics: {
    accuracy: {
      accuracy: number;
      precision: number;
      recall: number;
      f1_score: number;
      specificity: number;
      total_ratings: number;
    };
    content: {
      avg_helpfulness: number;
      avg_clarity: number;
      avg_confidence: number;
      total_ratings: number;
    };
  };
  summary: string;
}

export default function MetricsDisplay() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = async (days: number = 30) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/rating/metrics/?days=${days}&details=true`);
      const data = await response.json();
      
      if (response.ok) {
        setMetrics(data);
      } else {
        setError(data.error || "Failed to fetch metrics");
      }
    } catch (err) {
      setError("Network error: " + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;
  const formatRating = (value: number) => `${value.toFixed(1)}/5.0`;

  if (loading) {
    return (
      <Card className="w-full">
        <CardContent className="p-6">
          <div className="text-center">Loading metrics...</div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="w-full">
        <CardContent className="p-6">
          <div className="text-center text-red-600">
            Error: {error}
            <Button 
              onClick={() => fetchMetrics()} 
              className="ml-4"
              size="sm"
            >
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!metrics) {
    return (
      <Card className="w-full">
        <CardContent className="p-6">
          <div className="text-center">No metrics available</div>
        </CardContent>
      </Card>
    );
  }

  const { overall_metrics, total_ratings, period_days } = metrics;
  const { accuracy, content } = overall_metrics;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-800">Performance Metrics</h2>
        <div className="flex space-x-2">
          <Button 
            onClick={() => fetchMetrics(7)} 
            variant={period_days === 7 ? "default" : "outline"}
            size="sm"
          >
            7 days
          </Button>
          <Button 
            onClick={() => fetchMetrics(30)} 
            variant={period_days === 30 ? "default" : "outline"}
            size="sm"
          >
            30 days
          </Button>
          <Button 
            onClick={() => fetchMetrics(90)} 
            variant={period_days === 90 ? "default" : "outline"}
            size="sm"
          >
            90 days
          </Button>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <FileText className="w-5 h-5 text-blue-600" />
              <div>
                <p className="text-sm text-gray-600">Total Ratings</p>
                <p className="text-2xl font-bold">{total_ratings}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <TrendingUp className="w-5 h-5 text-green-600" />
              <div>
                <p className="text-sm text-gray-600">Accuracy</p>
                <p className="text-2xl font-bold">{formatPercentage(accuracy.accuracy)}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <BarChart3 className="w-5 h-5 text-purple-600" />
              <div>
                <p className="text-sm text-gray-600">F1 Score</p>
                <p className="text-2xl font-bold">{formatPercentage(accuracy.f1_score)}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Users className="w-5 h-5 text-orange-600" />
              <div>
                <p className="text-sm text-gray-600">Helpfulness</p>
                <p className="text-2xl font-bold">{formatRating(content.avg_helpfulness)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Legal Accuracy Metrics */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Legal Accuracy Metrics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Accuracy</span>
                <span className="font-semibold">{formatPercentage(accuracy.accuracy)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-green-600 h-2 rounded-full" 
                  style={{ width: `${accuracy.accuracy * 100}%` }}
                ></div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Precision</span>
                <span className="font-semibold">{formatPercentage(accuracy.precision)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full" 
                  style={{ width: `${accuracy.precision * 100}%` }}
                ></div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Recall</span>
                <span className="font-semibold">{formatPercentage(accuracy.recall)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-yellow-600 h-2 rounded-full" 
                  style={{ width: `${accuracy.recall * 100}%` }}
                ></div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">F1 Score</span>
                <span className="font-semibold">{formatPercentage(accuracy.f1_score)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-purple-600 h-2 rounded-full" 
                  style={{ width: `${accuracy.f1_score * 100}%` }}
                ></div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Specificity</span>
                <span className="font-semibold">{formatPercentage(accuracy.specificity)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-red-600 h-2 rounded-full" 
                  style={{ width: `${accuracy.specificity * 100}%` }}
                ></div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Content Quality Metrics */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Content Quality Metrics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Helpfulness</span>
                <span className="font-semibold">{formatRating(content.avg_helpfulness)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-orange-600 h-2 rounded-full" 
                  style={{ width: `${(content.avg_helpfulness / 5) * 100}%` }}
                ></div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Clarity</span>
                <span className="font-semibold">{formatRating(content.avg_clarity)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-indigo-600 h-2 rounded-full" 
                  style={{ width: `${(content.avg_clarity / 5) * 100}%` }}
                ></div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Confidence</span>
                <span className="font-semibold">{formatRating(content.avg_confidence)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-pink-600 h-2 rounded-full" 
                  style={{ width: `${(content.avg_confidence / 5) * 100}%` }}
                ></div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Summary Text */}
      {metrics.summary && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Analysis Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="whitespace-pre-wrap text-sm text-gray-700 font-mono">
              {metrics.summary}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
