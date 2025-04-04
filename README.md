# Fund Allocation Optimizer API

This web application provides a user-friendly interface and API for optimizing fund allocation across different banks and CD terms to maximize interest returns based on branch relationships and association data.

## Features

- Optimize fund allocation based on branch relationships and association data
- Allocate funds to maximize interest returns while respecting bank constraints
- Support for different banks with varying CD rates and terms
- Interactive data visualization
- Excel export functionality
- RESTful API for integration with other systems

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd fund-allocation-optimizer
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python app.py
   ```

The application will be available at http://localhost:5000.

## Usage

### Web Interface

1. Visit http://localhost:5000 in your web browser
2. Upload your Excel file with bank data
3. Enter your optimization parameters:
   - Total funds available
   - Investment duration preference
   - Multi-term investment option
   - CDARS products preference
   - Single bank restriction (optional)
4. Click "Optimize Allocation" to see the results
5. Download the results as an Excel file if needed

### API Endpoints

#### Optimize Fund Allocation

**Endpoint:** `POST /api/optimize`

**Content-Type:** `application/x-www-form-urlencoded` or `multipart/form-data`

**Parameters:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| branch_name | String | Name of the branch to optimize for | Yes |
| association_name | String | Name of the association to optimize for | Yes |

**Response:**

```json
{
  "success": true,
  "message": "Optimization completed successfully",
  "results": [
    {
      "Bank Name": "Bank A",
      "CD Term": "6 months",
      "Allocated Amount": 250000,
      "CD Rate": 2.5,
      "Expected Return": 6250
    },
    ...
  ],
  "summary": {
    "total_allocated": 1000000,
    "total_return": 25000,
    "weighted_avg_rate": 2.5,
    "total_funds": 1000000
  },
  "bank_count": 3,
  "term_count": 4
}
```

#### Get Banks, Branches, and Associations

**Endpoint:** `GET /api/banks`

**Response:**

```json
{
  "success": true,
  "banks": ["Bank A", "Bank B", "Bank C"]
}
```

**Endpoint:** `GET /api/branches`

**Response:**

```json
{
  "success": true,
  "branches": ["Branch 1", "Branch 2", "Branch 3"]
}
```

**Endpoint:** `GET /api/associations`

**Response:**

```json
{
  "success": true,
  "associations": ["Association 1", "Association 2", "Association 3"]
}
```

## Excel File Format

The application expects an Excel file with the following sheets:

1. **Bank Ranking**: Contains bank names, codes, and priority rankings
   - Required columns: "Bank Name", "Bank Code", "Priority"

2. **Bank Rates**: Contains CD terms and interest rates for various banks
   - Required columns: "Bank Name", "Bank Code", "CD Term", "CD Rate"
   - Optional columns: "CDARS Term", "CDARS Rate", "Special"

3. **Filter**: Contains constraints for the optimization process
   - Required columns: "Filter Name", "Min Value", "Max Value"

A sample template is available upon request.

## Environment Variables

The following environment variables can be set to configure the application:

- `PORT`: Port number for the application to listen on (default: 5000)
- `FLASK_DEBUG`: Set to "true" to enable debug mode (default: "false")
- `SECRET_KEY`: Secret key for session encryption (default: randomly generated)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 