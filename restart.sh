#!/bin/bash

# Emotion Recognition System - Simplified Restart Script
# Streamlined version with POSTER V2 as default

set -e

PROJECT_ROOT="/home/lleyt/WIL_project/refactor"
DASHBOARD_DIR="$PROJECT_ROOT/apps/dashboard"
PORT=8081

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to kill existing processes
kill_existing_processes() {
    echo -e "${YELLOW}Stopping existing processes...${NC}"
    
    # Kill processes on ports
    for port in 8081 5003 9091 8200; do
        if lsof -ti:$port >/dev/null 2>&1; then
            echo "Killing processes on port $port"
            lsof -ti:$port | xargs kill -9
            sleep 1
        fi
    done
    
    # Kill Label Studio specifically
    if lsof -ti:8200 >/dev/null 2>&1; then
        echo "Stopping Label Studio on port 8200"
        pkill -f "label-studio" 2>/dev/null || true
        sleep 1
    fi
    
    # Kill any remaining processes by name
    pkill -f "app\.py\|app_modular\.py\|dashboard_app\.py\|ml_service_app\.py" 2>/dev/null || true
    pkill -f "apps/ml_service" 2>/dev/null || true
    pkill -f "apps/label_studio_connector" 2>/dev/null || true
    
    echo -e "${GREEN}Existing processes stopped.${NC}"
}

# Function to start all services
start_services() {
    echo -e "${BLUE}Starting Emotion Recognition System...${NC}"
    echo -e "${GREEN}Configuration: POSTER V2 model with Action Unit detection${NC}"
    echo ""
    
    cd "$PROJECT_ROOT"
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Start ML engine service with POSTER V2
    echo "Starting ML engine service..."
    cd "$PROJECT_ROOT/apps/ml_service"
    USE_POSTER_V2=true ENABLE_AU_DETECTION=true python ml_service_app.py > "$PROJECT_ROOT/ml_service.log" 2>&1 &
    sleep 2
    
    # Start Label Studio connector
    echo "Starting Label Studio connector..."
    cd "$PROJECT_ROOT"
    python apps/label_studio_connector/app.py > "$PROJECT_ROOT/ml_backend.log" 2>&1 &
    sleep 2
    
    # Start dashboard
    echo "Starting dashboard..."
    cd "$DASHBOARD_DIR"
    python dashboard_app.py > "$PROJECT_ROOT/dashboard.log" 2>&1 &
    
    echo -e "${GREEN}All services started successfully!${NC}"
    sleep 3
}

# Function to show status
show_status() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════${NC}"
    echo -e "${BLUE}         System Status${NC}"
    echo -e "${BLUE}═══════════════════════════════════════${NC}"
    echo ""
    
    # Check each service
    if lsof -ti:$PORT >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Dashboard${NC}              Running on port $PORT"
    else
        echo -e "${RED}✗ Dashboard${NC}              Not running"
    fi
    
    if lsof -ti:5003 >/dev/null 2>&1; then
        echo -e "${GREEN}✓ ML Service${NC}             Running on port 5003"
    else
        echo -e "${RED}✗ ML Service${NC}             Not running"
    fi
    
    if lsof -ti:9091 >/dev/null 2>&1; then
        echo -e "${GREEN}✓ LS Connector${NC}           Running on port 9091"
    else
        echo -e "${RED}✗ LS Connector${NC}           Not running"
    fi
    
    if lsof -ti:8200 >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Label Studio${NC}           Running on port 8200"
    else
        echo -e "${RED}✗ Label Studio${NC}           Not running"
    fi
    
    echo ""
    echo -e "${BLUE}Service URLs:${NC}"
    echo "  • Dashboard:     http://localhost:$PORT"
    echo "  • Label Studio:  http://localhost:8200"
    echo "  • ML Status:     http://localhost:5003/status"
    echo "  • LS Connector:  http://localhost:9091/health"
    echo ""
}

# Function to show logs
show_logs() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════${NC}"
    echo -e "${BLUE}         Recent Logs${NC}"
    echo -e "${BLUE}═══════════════════════════════════════${NC}"
    echo ""
    
    if [ -f "$PROJECT_ROOT/dashboard.log" ]; then
        echo -e "${YELLOW}Dashboard (last 5 lines):${NC}"
        tail -5 "$PROJECT_ROOT/dashboard.log"
        echo ""
    fi
    
    if [ -f "$PROJECT_ROOT/ml_service.log" ]; then
        echo -e "${YELLOW}ML Service (last 5 lines):${NC}"
        tail -5 "$PROJECT_ROOT/ml_service.log"
        echo ""
    fi
    
    if [ -f "$PROJECT_ROOT/ml_backend.log" ]; then
        echo -e "${YELLOW}Label Studio Connector (last 5 lines):${NC}"
        tail -5 "$PROJECT_ROOT/ml_backend.log"
        echo ""
    fi
}

# Main menu
show_menu() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════${NC}"
    echo -e "${BLUE}    Emotion Recognition System${NC}"
    echo -e "${BLUE}═══════════════════════════════════════${NC}"
    echo ""
    echo "1) Start System"
    echo "2) Show Status"
    echo "3) View Logs"
    echo "4) Stop All Services"
    echo "5) Exit"
    echo ""
    echo -n "Select option (1-5): "
}

# Main execution
main() {
    while true; do
        show_menu
        read choice
        
        case $choice in
            1)
                echo ""
                kill_existing_processes
                start_services
                show_status
                echo -e "${YELLOW}System is ready! Access at: http://localhost:$PORT${NC}"
                ;;
            2)
                show_status
                ;;
            3)
                show_logs
                ;;
            4)
                echo ""
                kill_existing_processes
                echo -e "${GREEN}All services stopped.${NC}"
                ;;
            5)
                echo -e "${GREEN}Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option. Please select 1-5.${NC}"
                ;;
        esac
        
        if [ "$choice" != "5" ]; then
            echo ""
            echo -n "Press Enter to continue..."
            read
        fi
    done
}

# Run main function
main