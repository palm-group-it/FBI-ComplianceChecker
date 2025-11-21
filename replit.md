# Overview

This is a compliance web application built with Python and Streamlit that helps identify sales agents whose insurer mix significantly deviates from the company-wide baseline. The application processes Excel exports containing contract data and performs statistical analysis to flag agents with unusual insurer distribution patterns within specific lines of business.

The primary use case is for compliance teams to quickly upload Excel reports and identify potential risks or anomalies in agent behavior regarding insurer selection.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Application Framework
**Decision**: Single-file Streamlit application (app.py)  
**Rationale**: Streamlit provides rapid development with built-in UI components, perfect for internal compliance tools. The single-file approach minimizes deployment complexity and keeps the codebase simple for non-technical users who may need to maintain it.  
**Trade-offs**: Limited scalability and customization compared to full web frameworks, but appropriate for the internal tool use case.

## Data Processing
**Decision**: Pandas for all data manipulation and statistical calculations  
**Rationale**: Pandas excels at tabular data operations and integrates seamlessly with Excel file formats. The domain involves analyzing contract counts grouped by various dimensions (agent, line of business, insurer), which aligns perfectly with pandas' GroupBy functionality.  
**Implementation**: 
- Excel file parsing with multi-sheet support
- Column mapping from Hungarian business terms to English technical names
- Aggregate calculations for baseline distributions and agent-specific distributions
- Missing value handling with explicit labels

## Statistical Analysis Method
**Decision**: Count-based deviation analysis rather than premium or value-based  
**Rationale**: Each contract row represents equal weight (count=1), making the analysis simpler and more transparent. The core logic compares agent-level insurer distributions against company-wide baseline distributions within each line of business.

**Algorithm**:
1. Calculate baseline insurer mix per line of business across all agents
2. Calculate each agent's insurer mix per line of business
3. Compute percentage point deviations between agent and baseline
4. Flag deviations exceeding configurable threshold

## Data Model
**Schema**: Three required columns from Excel input
- `UkKodja1` (agent_id): Sales agent identifier
- `Megnevezés` (line_of_business): Insurance product category/sector
- `RövidNév` (insurer): Insurance company short name

**Constraints**: 
- One row = one contract
- No temporal filtering (Excel export pre-filtered for correct period)
- Missing values handled as explicit "(Missing)" category
- Multi-dimensional grouping (agent × line_of_business × insurer)

## User Interface Design
**Decision**: Wide layout with file upload and sheet selection  
**Rationale**: Compliance users need to see tabular results clearly. Wide layout maximizes horizontal space for result tables. Interactive sheet selection handles variability in Excel export formats where the target sheet name may differ.

# External Dependencies

## Core Libraries
- **Streamlit**: Web application framework providing UI components and session management
- **Pandas**: Data manipulation, Excel I/O, and statistical computations
- **io**: In-memory file handling for uploaded Excel files

## Data Format
- **Input**: Excel (.xlsx) files with multiple sheets
- **Expected sheet**: "DosszieAdatok282 - eredeti" (but user-selectable)
- **Output**: In-app tabular display of flagged deviations

## No External Services
The application runs entirely locally without databases, APIs, or cloud services. All computation happens in-memory during the user session.