package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"sort"
	"strconv"

	"github.com/montanaflynn/stats" // Import the montanaflynn/stats package
)

// DataPoint represents a single row in the CSV file.
type DataPoint struct {
	Dataset string
	X       float64
	Y       float64
}

func main() {
	filePath := "anscombe.csv"

	// 1. Read and parse the CSV data
	data, err := readCSVData(filePath)
	if err != nil {
		log.Fatalf("Error reading or parsing CSV file '%s': %v", filePath, err)
	}

	// 2. Group data by the 'Dataset' column
	groupedData := groupDataByDataset(data)

	// Get dataset keys and sort them for consistent output order
	datasets := make([]string, 0, len(groupedData))
	for k := range groupedData {
		datasets = append(datasets, k)
	}
	sort.Strings(datasets) // Sort keys: I, II, III, IV

	fmt.Println("Linear Regression Results for Anscombe's Quartet (using montanaflynn/stats):")
	fmt.Println("-------------------------------------------------------------------------")

	// 3. Perform linear regression for each group and print results
	for _, datasetName := range datasets {
		points := groupedData[datasetName]

		// calculateLinearRegression now handles the minimum points check internally via the library
		slope, intercept, err := calculateLinearRegression(points)
		if err != nil {
			// The library returns specific errors like stats.EmptyInputErr or stats.NotEnoughElementsErr
			fmt.Printf("Dataset: %s | Error calculating regression: %v\n", datasetName, err)
			continue
		}

		// Print results in the desired format
		fmt.Printf("Dataset: %s | Slope: %.3f | Intercept: %.3f\n",
			datasetName, slope, intercept)
	}
	fmt.Println("-------------------------------------------------------------------------")
}

// readCSVData opens, reads, and parses the CSV file into a slice of DataPoint structs.
// (This function remains the same as the previous version)
func readCSVData(filePath string) ([]DataPoint, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.TrimLeadingSpace = true

	// Read header row
	_, err = reader.Read()
	if err != nil {
		if err == io.EOF {
			return nil, fmt.Errorf("file is empty or contains only header")
		}
		return nil, fmt.Errorf("failed to read header row: %w", err)
	}

	var dataPoints []DataPoint

	// Read remaining rows
	for {
		record, err := reader.Read()
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("error reading CSV record: %w", err)
		}

		if len(record) != 3 {
			log.Printf("Skipping invalid record (expected 3 columns, got %d): %v", len(record), record)
			continue
		}

		xVal, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			log.Printf("Skipping record due to invalid X value '%s': %v", record[1], err)
			continue
		}

		yVal, err := strconv.ParseFloat(record[2], 64)
		if err != nil {
			log.Printf("Skipping record due to invalid Y value '%s': %v", record[2], err)
			continue
		}

		dp := DataPoint{
			Dataset: record[0],
			X:       xVal,
			Y:       yVal,
		}
		dataPoints = append(dataPoints, dp)
	}

	if len(dataPoints) == 0 {
		return nil, fmt.Errorf("no valid data records found in the file")
	}

	return dataPoints, nil
}

// groupDataByDataset groups the slice of DataPoint structs into a map.
// (This function remains the same as the previous version)
func groupDataByDataset(data []DataPoint) map[string][]DataPoint {
	groups := make(map[string][]DataPoint)
	for _, point := range data {
		groups[point.Dataset] = append(groups[point.Dataset], point)
	}
	return groups
}

func calculateLinearRegression(points []DataPoint) (float64, float64, error) {
	var xVals, yVals []float64
	for _, p := range points {
		xVals = append(xVals, p.X)
		yVals = append(yVals, p.Y)
	}

	xMean, err := stats.Mean(xVals)
	if err != nil {
		return 0, 0, err
	}
	yMean, err := stats.Mean(yVals)
	if err != nil {
		return 0, 0, err
	}

	numerator := 0.0
	denominator := 0.0
	for i := range xVals {
		numerator += (xVals[i] - xMean) * (yVals[i] - yMean)
		denominator += (xVals[i] - xMean) * (xVals[i] - xMean)
	}

	slope := numerator / denominator
	intercept := yMean - slope*xMean

	return slope, intercept, nil
}
