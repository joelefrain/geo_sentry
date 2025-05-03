<<<<<<< HEAD
CREATE TABLE company (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(20) UNIQUE,
    name VARCHAR(100) NOT NULL,
    logo_path VARCHAR(255),
    state BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
--
CREATE TABLE project (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(20) UNIQUE,
    name VARCHAR(100) NOT NULL,
    company_code VARCHAR(20) NOT NULL,
    utm_zone INTEGER CHECK (utm_zone BETWEEN 1 AND 60),
    utm_hemisphere VARCHAR(1) CHECK (utm_hemisphere IN ('N', 'S')),
    state BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (company_code) REFERENCES company (code)
);
--
CREATE TABLE component (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(20) UNIQUE,
    name VARCHAR(100) NOT NULL,
    project_code VARCHAR(20) NOT NULL,
    description TEXT,
    geometry_path VARCHAR(255) NOT NULL,
    state BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_code) REFERENCES project (code)
);
--
CREATE TABLE sector (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(20) UNIQUE,
    name VARCHAR(100) NOT NULL,
    component_code VARCHAR(20) NOT NULL,
    description TEXT,
    state BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (component_code) REFERENCES component (code)
);
--
CREATE TABLE sensor (
    code VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    sector_code VARCHAR(20) NOT NULL,
    sensor_type VARCHAR(5) NOT NULL,
    alert_level JSON,
    east DECIMAL(12, 6) NOT NULL,
    north DECIMAL(12, 6) NOT NULL,
    elevation DECIMAL(10, 4) NOT NULL,
    description TEXT,
    state BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sector_code) REFERENCES sector (code)
);
--
CREATE TABLE sensor_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sensor_code VARCHAR(20) NOT NULL,
    time TIMESTAMP NOT NULL,
    east DECIMAL(12, 6) NOT NULL,
    north DECIMAL(12, 6) NOT NULL,
    elevation DECIMAL(10, 4) NOT NULL,
    base_line BOOLEAN DEFAULT FALSE,
    processed_data JSON,
    is_processed BOOLEAN DEFAULT FALSE,
    is_shown BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sensor_code) REFERENCES sensor (code)
);
--
CREATE TRIGGER update_company_timestamp 
AFTER UPDATE ON company 
FOR EACH ROW
BEGIN
    UPDATE company SET modified_at = CURRENT_TIMESTAMP WHERE id = old.id;
END;
--
CREATE TRIGGER update_project_timestamp 
AFTER UPDATE ON project 
FOR EACH ROW
BEGIN
    UPDATE project SET modified_at = CURRENT_TIMESTAMP WHERE id = old.id;
END;
--
CREATE TRIGGER update_component_timestamp 
AFTER UPDATE ON component 
FOR EACH ROW
BEGIN
    UPDATE component SET modified_at = CURRENT_TIMESTAMP WHERE id = old.id;
END;
--
CREATE TRIGGER update_sector_timestamp 
AFTER UPDATE ON sector 
FOR EACH ROW
BEGIN
    UPDATE sector SET modified_at = CURRENT_TIMESTAMP WHERE id = old.id;
END;
--
CREATE TRIGGER update_sensor_timestamp 
AFTER UPDATE ON sensor 
FOR EACH ROW
BEGIN
    UPDATE sensor SET modified_at = CURRENT_TIMESTAMP WHERE code = old.code;
END;
--
CREATE TRIGGER update_sensor_data_timestamp 
AFTER UPDATE ON sensor_data 
FOR EACH ROW
BEGIN
    UPDATE sensor_data SET modified_at = CURRENT_TIMESTAMP WHERE id = old.id;
=======
CREATE TABLE company (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(20) UNIQUE,
    name VARCHAR(100) NOT NULL,
    logo_path VARCHAR(255),
    state BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
--
CREATE TABLE project (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(20) UNIQUE,
    name VARCHAR(100) NOT NULL,
    company_code VARCHAR(20) NOT NULL,
    utm_zone INTEGER CHECK (utm_zone BETWEEN 1 AND 60),
    utm_hemisphere VARCHAR(1) CHECK (utm_hemisphere IN ('N', 'S')),
    state BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (company_code) REFERENCES company (code)
);
--
CREATE TABLE component (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(20) UNIQUE,
    name VARCHAR(100) NOT NULL,
    project_code VARCHAR(20) NOT NULL,
    description TEXT,
    geometry_path VARCHAR(255) NOT NULL,
    state BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_code) REFERENCES project (code)
);
--
CREATE TABLE sector (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(20) UNIQUE,
    name VARCHAR(100) NOT NULL,
    component_code VARCHAR(20) NOT NULL,
    description TEXT,
    state BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (component_code) REFERENCES component (code)
);
--
CREATE TABLE sensor (
    code VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    sector_code VARCHAR(20) NOT NULL,
    sensor_type VARCHAR(5) NOT NULL,
    alert_level JSON,
    east DECIMAL(12, 6) NOT NULL,
    north DECIMAL(12, 6) NOT NULL,
    elevation DECIMAL(10, 4) NOT NULL,
    description TEXT,
    state BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sector_code) REFERENCES sector (code)
);
--
CREATE TABLE sensor_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sensor_code VARCHAR(20) NOT NULL,
    time TIMESTAMP NOT NULL,
    east DECIMAL(12, 6) NOT NULL,
    north DECIMAL(12, 6) NOT NULL,
    elevation DECIMAL(10, 4) NOT NULL,
    base_line BOOLEAN DEFAULT FALSE,
    processed_data JSON,
    is_processed BOOLEAN DEFAULT FALSE,
    is_shown BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sensor_code) REFERENCES sensor (code)
);
--
CREATE TRIGGER update_company_timestamp 
AFTER UPDATE ON company 
FOR EACH ROW
BEGIN
    UPDATE company SET modified_at = CURRENT_TIMESTAMP WHERE id = old.id;
END;
--
CREATE TRIGGER update_project_timestamp 
AFTER UPDATE ON project 
FOR EACH ROW
BEGIN
    UPDATE project SET modified_at = CURRENT_TIMESTAMP WHERE id = old.id;
END;
--
CREATE TRIGGER update_component_timestamp 
AFTER UPDATE ON component 
FOR EACH ROW
BEGIN
    UPDATE component SET modified_at = CURRENT_TIMESTAMP WHERE id = old.id;
END;
--
CREATE TRIGGER update_sector_timestamp 
AFTER UPDATE ON sector 
FOR EACH ROW
BEGIN
    UPDATE sector SET modified_at = CURRENT_TIMESTAMP WHERE id = old.id;
END;
--
CREATE TRIGGER update_sensor_timestamp 
AFTER UPDATE ON sensor 
FOR EACH ROW
BEGIN
    UPDATE sensor SET modified_at = CURRENT_TIMESTAMP WHERE code = old.code;
END;
--
CREATE TRIGGER update_sensor_data_timestamp 
AFTER UPDATE ON sensor_data 
FOR EACH ROW
BEGIN
    UPDATE sensor_data SET modified_at = CURRENT_TIMESTAMP WHERE id = old.id;
>>>>>>> 118aabc (update | Independizacion del locale del sistema operativo)
END;