"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { CheckCircle, Star, XCircle } from "lucide-react";
import { useState } from "react";

interface RatingData {
  correctness: boolean;
  confidence: number;
  helpfulness: number;
  clarity: number;
  comment: string;
  user_id: string;
}

interface RatingComponentProps {
  query: string;
  response: string;
  caseId?: string;
  onRatingSubmitted: () => void;
}

export default function RatingComponent({ 
  query, 
  response, 
  caseId, 
  onRatingSubmitted 
}: RatingComponentProps) {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [ratingData, setRatingData] = useState<RatingData>({
    correctness: false,
    confidence: 3,
    helpfulness: 3,
    clarity: 3,
    comment: "",
    user_id: ""
  });

  const handleSubmit = async () => {
    if (!ratingData.correctness) {
      alert("Please provide a correctness rating");
      return;
    }

    setIsSubmitting(true);
    
    try {
      const apiResponse = await fetch("http://127.0.0.1:8000/api/rating/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          response,
          case_id: caseId || "",
          ...ratingData
        }),
      });

      if (apiResponse.ok) {
        alert("Rating submitted successfully!");
        onRatingSubmitted();
        // Reset form
        setRatingData({
          correctness: false,
          confidence: 3,
          helpfulness: 3,
          clarity: 3,
          comment: "",
          user_id: ""
        });
      } else {
        const error = await apiResponse.json();
        alert(`Error: ${error.message || "Failed to submit rating"}`);
      }
    } catch (error) {
      console.error("Error submitting rating:", error);
      alert("Failed to submit rating. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const StarRating = ({ 
    value, 
    onChange, 
    label 
  }: { 
    value: number; 
    onChange: (value: number) => void; 
    label: string;
  }) => (
    <div className="space-y-1">
      <label className="text-xs font-medium text-gray-600">{label}</label>
      <div className="flex items-center space-x-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            type="button"
            onClick={() => onChange(star)}
            className={`p-0.5 ${
              star <= value ? "text-yellow-400" : "text-gray-300"
            } hover:text-yellow-400 transition-colors`}
          >
            <Star className="w-4 h-4 fill-current" />
          </button>
        ))}
        <span className="ml-2 text-xs text-gray-500">{value}/5</span>
      </div>
    </div>
  );

  return (
    <Card className="border border-gray-200 shadow-sm bg-white">
      <CardContent className="space-y-4 p-4">
          {/* Correctness Rating */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Was this correct?</span>
              <div className="flex items-center space-x-2">
                <Button
                  variant={ratingData.correctness ? "default" : "outline"}
                  size="sm"
                  onClick={() => setRatingData(prev => ({ ...prev, correctness: true }))}
                  className={`text-xs ${ratingData.correctness ? 'bg-green-600 text-white hover:bg-green-700' : 'text-green-600 border-green-600 hover:bg-green-50'}`}
                >
                  <CheckCircle className="w-3 h-3 mr-1" />
                  Yes
                </Button>
                <Button
                  variant={!ratingData.correctness ? "default" : "outline"}
                  size="sm"
                  onClick={() => setRatingData(prev => ({ ...prev, correctness: false }))}
                  className={`text-xs ${!ratingData.correctness ? 'bg-red-600 text-white hover:bg-red-700' : 'text-red-600 border-red-600 hover:bg-red-50'}`}
                >
                  <XCircle className="w-3 h-3 mr-1" />
                  No
                </Button>
              </div>
            </div>
          </div>

          {/* Star Ratings - Each in separate row */}
          <div className="space-y-4">
            <StarRating
              value={ratingData.confidence}
              onChange={(value) => setRatingData(prev => ({ ...prev, confidence: value }))}
              label="Confidence (1-5)"
            />
            <StarRating
              value={ratingData.helpfulness}
              onChange={(value) => setRatingData(prev => ({ ...prev, helpfulness: value }))}
              label="Helpfulness (1-5)"
            />
            <StarRating
              value={ratingData.clarity}
              onChange={(value) => setRatingData(prev => ({ ...prev, clarity: value }))}
              label="Clarity (1-5)"
            />
          </div>

          {/* Comment */}
          <div className="space-y-1">
            <label className="text-xs font-medium text-gray-600">
              Additional feedback (optional)
            </label>
            <Input
              value={ratingData.comment}
              onChange={(e) => setRatingData(prev => ({ ...prev, comment: e.target.value }))}
              placeholder="Any comments about this response..."
              className="text-xs h-8"
            />
          </div>

          {/* User ID */}
          <div className="space-y-1">
            <label className="text-xs font-medium text-gray-600">
              Your identifier (optional)
            </label>
            <Input
              value={ratingData.user_id}
              onChange={(e) => setRatingData(prev => ({ ...prev, user_id: e.target.value }))}
              placeholder="e.g., student123, expert_lawyer"
              className="text-xs h-8"
            />
          </div>

          {/* Submit Button */}
          <div className="flex justify-end space-x-2 pt-1">
            <Button
              size="sm"
              onClick={handleSubmit}
              disabled={isSubmitting}
              className="bg-blue-600 hover:bg-blue-700 text-xs"
            >
              {isSubmitting ? "Submitting..." : "Submit Rating"}
            </Button>
          </div>
      </CardContent>
    </Card>
  );
}
