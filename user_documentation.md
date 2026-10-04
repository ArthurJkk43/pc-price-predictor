# PC Build Price Estimator User Guide

The PC Build Price Estimator forecasts the price of selected computer components approximately 180 days into the future. It uses pre-trained machine-learning models, historical pricing data, and component specifications to produce an estimated price in US dollars.

This guide explains how to start the application, enter component data, generate a forecast, interpret the results, and resolve common problems.

> **Important:** Forecasts are estimates, not guaranteed future prices. Use them for planning and comparison rather than purchasing or financial decisions.

## Contents

- [System overview](#system-overview)
- [Supported components](#supported-components)
- [Requirements](#requirements)
- [Starting the application](#starting-the-application)
- [Generating a price forecast](#generating-a-price-forecast)
- [Entering component data](#entering-component-data)
- [Understanding the results](#understanding-the-results)
- [Improving forecast quality](#improving-forecast-quality)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [References](#references)

## System overview

The application provides a browser-based interface for estimating future PC component prices. When a user selects a component, the interface displays the fields required by that component's prediction model. Submitted values are validated and converted into the format expected by the stored model. The server then generates a forecast without retraining the model.

The application uses Flask and Jinja2 for the web interface. Its prediction pipeline may also rely on pandas, scikit-learn, CatBoost, Keepa, and supporting datasets or lookup files.

### Intended users

The system is suitable for:

- PC builders planning a future purchase
- Students and researchers exploring hardware price trends
- Technology enthusiasts comparing possible configurations
- Users testing how component specifications affect estimated prices

Basic familiarity with hardware specifications is helpful. Depending on the selected category, the application may request information such as model name, chipset, memory capacity, memory type, architecture, launch year, storage capacity, dimensions, or power consumption.

## Supported components

The interface includes the following categories:

| Component | Examples of relevant specifications |
| --- | --- |
| GPU | Chipset, board partner, VRAM, memory type, architecture, current price |
| CPU | Processor model or chipset and current price |
| SSD | Capacity, interface, form factor, and related storage specifications |
| RAM | Capacity, memory generation, speed, and related memory specifications |
| Motherboard | Chipset, socket, form factor, and supported standards |
| Monitor | Size, resolution, refresh rate, and panel-related specifications |
| PSU | Rated power, efficiency, and form factor |
| Case | Size, form factor, and compatibility-related specifications |
| CPU cooler | Cooler type, dimensions, and compatibility-related specifications |

Model availability and forecast quality may differ by category. A category with more consistent historical data and clearly defined specifications may produce more reliable results than one supported by sparse or mainly transaction-based data.

## Requirements

Before starting the application, confirm that the project environment contains:

- Python 3.11 or the Python version configured for the project
- Flask
- Jinja2
- pandas
- scikit-learn
- CatBoost
- Playwright, if required by the data or testing workflow
- the Python Keepa library wrapper, if required by the project
- trained model files
- datasets, lookup tables, templates, and other supporting artifacts used by the application

Run the application from its project directory. The relative paths used by `app.py` may fail if the program is started from another location.

## Starting the application

1. Open a terminal in the project directory containing `app.py`.
2. Activate the project's Python environment, if one is used.
3. Start the Flask application:

   ```bash
   python app.py
   ```

4. Wait for the server to finish loading the stored models and supporting files.
5. Open the localhost address printed in the terminal.

If the page does not load, keep the terminal open and review its messages. They usually identify a missing dependency, model file, dataset, template, or incorrect working directory.

## Generating a price forecast

![PC Build Price Estimator interface](user_documentation_assets/system-interface.png)

*Figure 1. Main interface with GPU selected and a generated 180-day forecast.*

1. In the **Components** panel, select each part you want to include.
2. Complete the fields displayed for every selected component.
3. Use the supplied dropdown options when available.
4. Enter the current price in USD if the field is available and you know the value.
5. Review the selected models and specifications for consistency.
6. Select **Predict prices**.
7. Read the forecast for each component in the **Results** section.
8. Review the displayed total when forecasting more than one component.

Selecting a category activates the prediction model associated with that component. Because each model expects different data, the visible form fields change by category.

## Entering component data

Use specifications from the product page, manufacturer documentation, or another reliable source whenever possible.

### Preset values

Some fields provide dropdown lists populated from stored reference data. Preset values reduce formatting errors and help the application map the input to values recognized by the model.

Choose the option that matches the actual product. A preset or default value is a convenience, not proof that it is correct for the selected component.

### Optional values

Fields marked as optional may be left empty. The model can impute some missing values, but a forecast based on accurate specifications is generally more useful than one based on inferred data.

### Current price

When available, enter the component's current market price in USD. Use a realistic price from the same market context as the rest of the input. Avoid mixing prices from different regions, currencies, or market conditions unless they have been converted consistently.

### Consistent specifications

All values should describe the same real or plausible product. Do not combine a chipset from one model with the memory type, capacity, architecture, or release period of another.

![Example of inconsistent GPU input](user_documentation_assets/invalid-input-example.png)

*Figure 2. Example of inconsistent GPU values. The selected chipset, board partner, VRAM, memory type, and architecture do not form a reliable product description.*

If an exact value is unavailable, choose the closest supported option only when the combination remains technically plausible. Do not enter an extreme value solely to obtain a different forecast.

## Understanding the results

The results panel displays an estimated price for each selected component. When several components are included, it also displays their estimated total.

The forecast represents the model's estimate for approximately 180 days after the current observation point. It is based on patterns found in historical data and the specifications supplied through the form.

Interpret the result as a planning estimate:

- It is not a guaranteed retail price.
- It may differ from prices shown by a specific store or region.
- It is most useful for comparing plausible scenarios under similar assumptions.
- Its reliability depends on the selected category, training data, and input quality.

## Improving forecast quality

For a more meaningful estimate:

1. Use specifications from a single, identifiable product.
2. Enter accurate values instead of relying on defaults when the correct data is available.
3. Keep units, currencies, and market context consistent.
4. Avoid unusual combinations that are unlikely to exist in the real market.
5. Treat forecasts for products unlike the model's training examples with additional caution.
6. Compare the forecast with current market information before acting on it.

The model is more likely to produce a reasonable estimate when the submitted product resembles examples represented in its training data.

## Troubleshooting

### Input and prediction problems

| Problem | Likely cause | Recommended action |
| --- | --- | --- |
| The form does not submit | A required field is empty or contains an invalid value | Review every visible field and use a preset option where available |
| The result looks unrealistic | Specifications are inconsistent or far outside typical training examples | Verify that all values describe the same product and replace placeholder values |
| No result appears | Validation failed or the prediction request produced a server error | Review the form, then check the terminal or application log |
| A component field is missing | The wrong category is selected or the current model does not use that field | Confirm the selected component and use only the fields presented by the interface |
| A forecast fails for one category | Its model file, lookup data, or preprocessing configuration is unavailable | Confirm that the category's required artifacts are present in the expected paths |

### Application startup problems

| Problem | Recommended action |
| --- | --- |
| The localhost page does not open | Confirm that `app.py` is running and use the address printed in the terminal |
| A Python module cannot be imported | Install or activate the project's configured dependencies |
| A model or data file cannot be found | Start the application from the project directory and verify the expected file locations |
| The server returns an internal error | Review the terminal output for the failing file, dependency, validation step, or preprocessing operation |
| The interface loads but prediction fails | Confirm that the trained models, datasets, lookup tables, and templates are available and compatible |

When reporting a problem, include the selected component, the values entered, the visible error message, and the relevant terminal log. Do not include passwords, access tokens, or other secrets.

## Known limitations

### Forecast uncertainty

The models learn from historical observations. They cannot reliably anticipate events that are absent from the training data, including sudden demand changes, supply shortages, tax or tariff changes, regional pricing shifts, political events, or major manufacturer decisions.

### Uneven category support

The amount and quality of available data differ across component categories. Accuracy may therefore vary between GPUs, CPUs, storage devices, memory, and other hardware.

### Out-of-distribution products

Forecasts become less dependable when a product differs substantially from examples used during model training. Newly released components and unusual specification combinations are common sources of uncertainty.

### Prototype interface

The application emphasizes forecasting functionality over production-level interface features. User accounts, automatic device detection, and other commercial product features are outside the prototype's scope.

### No model training during use

The application loads pre-trained models when it starts. Submitting a form generates a prediction but does not retrain or update a model.

## References

- [CatBoost documentation](https://catboost.ai/docs/en/)
- [Flask templating documentation](https://flask.palletsprojects.com/en/stable/templating/)
- [Jinja documentation](https://jinja.palletsprojects.com/en/stable/)
- [Keepa Python API documentation](https://keepaapi.readthedocs.io/en/stable/api_methods.html)
- [pandas documentation](https://pandas.pydata.org/docs/)
- [Playwright documentation](https://playwright.dev/docs/intro)
- [Python 3.11 documentation](https://docs.python.org/3.11/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [The semiconductor decade McKinsey](https://www.mckinsey.com/industries/semiconductors/our-insights/the-semiconductor-decade-a-trillion-dollar-industry)

