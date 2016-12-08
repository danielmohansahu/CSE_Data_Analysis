//////////////////////////////// INITIALIZATION ////////////////////////////////

// Load patient information from inputForm:
var input = {
  "Body Height": sessionStorage.height,
  "Body Weight": sessionStorage.weight,
  "Age": sessionStorage.age,
  "Creatinine": sessionStorage.creatinine,
  "Calcium": sessionStorage.calcium,
  "Systolic BP": sessionStorage.blood_pressure,
  "Total Cholesterol": sessionStorage.cholesterol,
  "Body Mass Index": sessionStorage.bmi,
  "Potassium": sessionStorage.potassium,
  "Female": sessionStorage.female,
  "Male": sessionStorage.male,
  "Glucose": sessionStorage.glucose,
  "Sodium":  sessionStorage.sodium,
  "LDL Cholesterol": sessionStorage.ldl,
  "HDL Cholesterol": sessionStorage.hdl,
  "Smoker": sessionStorage.smoker,
  "Hemoglobin A1c": sessionStorage.hemoglobin_a1c}

var importance = {
  "Age": 0.276700,
  "Body Height": 0.255053,
  "Female": 0.170558,
  "Body Weight": 0.135051,
  "Body Mass Index": 0.066687,
  "Creatinine": 0.031676,
  "Total Cholesterol": 0.013504,
  "HDL Cholesterol": 0.011664,
  "Hemoglobin A1c": 0.008698,
  "Potassium": 0.008619,
  "Smoker": 0.004763,
  "LDL Cholesterol": 0.004647,
  "Systolic BP": 0.003297,
  "Calcium": 0.003201,
  "Urea Nitrogen": 0.002127,
  "Carbon Dioxide": 0.002091,
  "Diastolic BP": 0.000926,
  "Glucose": 0.000513,
  "Triglycerides": 0.000226,
  "Male": 0.000000,
  "Chloride": 0.000000,
  "Sodium": 0.000000 }

var print_recommend = 0;
sessionStorage.recommendation = NaN

// Population numbers (hardcoded from generated synthea data)
var num_healthy = 3007,
    num_sick = 3009,
    total_healthy = 123000 - num_sick;  // Rough estimate!

// true: highlight user biometric path; false: don't highlight
var highlight_flag = true,
    highlighted = false,    // Used to only update once
    recommended = false;

// Convert string to boolean
if (sessionStorage.recommend_flag == "true") {
  var recommend_flag = true;
} else {
  var recommend_flag = false;
}

var margin = {top: 20, right: 120, bottom: 20, left: 180},
    width = 1400 - margin.right - margin.left,
//    height = 700 - margin.top - margin.bottom; // here
    height = 800 - margin.top - margin.bottom; // here

var i = 0,
    counter = 0,
    root;

// Tree
var tree = d3.layout.tree()
    .size([height, width]);

// Connecting Lines
var diagonal = d3.svg.diagonal()
    .projection(function(d) { return [d.y, d.x]; });

// SVG Body
var svg = d3.select(sessionStorage.location).append("svg")
    .attr("width", width + margin.right + margin.left)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var legend_data = [ "Original path", "Recommended path", 
"False branch", "True branch", "True branch along user path", "False branch along user path"];



var legend = svg.append("g")
             .attr("class", "legend")
             .attr("height", 100)
             .attr("width", 100)
             .attr("transform", "translate(-20,50)");

legend.selectAll('rect')
    .data(legend_data)
    .enter()
    .append('rect')
    .attr("x",-100)
    .attr("y", function(d,i) { return height -80 - i*20;})
    .attr("height", 7.5)
    .attr("width", 25)
    .style("fill", function(d){

        if (d == "False branch" || d == "False branch along user path") {
            return "#800000";
        } else if (d == "True branch" || d == "True branch along user path") {
            return "#006600";
        } else if (d == "Recommended path") {
            return "#cc5200";
        } else if (d == "Original path") {
            return "#000000";
        }
        else{
        	return "#006600";
        }
    })
    .style("opacity", 0.4)
    .style("stroke", "#f2f2f2")
    .style("stroke-width", function(d){
        if (d == "False branch" || d == "True branch"){
        	return 5;
        }
        else return 0;

    });



legend.selectAll('text')
    .data(legend_data)
    .enter()
    .append("text")
    .attr("x", -70)
    .attr("y", function(d, i){ return height - 71 - (i*20);})
    .text( function(d) {
        if (d.split(" ")[0] == "Recommended") {
            return "Recommended path (click 'Recommendations')";
        } else {
            return d;
        } });



//////////////////////////////// UPDATE SYSTEM ////////////////////////////////

d3.json("../tree.json", function(error, flare) {
  if (error) throw error;

  root = flare;
  root.x0 = height / 2;
  root.y0 = 0;

  root.children.forEach(collapse);
  update(root);

    // d3.select(sessionStorage.paraLocation).append("h3").html(sessionStorage.recommendation).style("text-align","center"); // here
    if (print_recommend == 1){
        d3.select(sessionStorage.paraLocation).append("h3").html(sessionStorage.recommendation).style("text-align","center"); // here
    } else if (print_recommend == 2) {
        d3.select(sessionStorage.paraLocation).append("h3").html("No recommendations. You are healthy.").style("text-align","center"); // here
    }
});

//////////////////////////////// FUNCTIONS ////////////////////////////////

function update(source) {

  // Compute the new tree layout.
  var nodes = tree.nodes(root).reverse(),
      links = tree.links(nodes);

  // Normalize for fixed-depth.
  nodes.forEach(function(d) { d.y = d.depth * 125; });

  // Update the nodes…
  var node = svg.selectAll("g.node")
      .data(nodes, function(d) { return d.id || (d.id = ++i); });

  // Enter any new nodes at the parent's previous position.
  var nodeEnter = node.enter().append("g")
      .attr("class", "node")
      .on("click", click);

  nodeEnter.append("circle")

  nodeEnter.append("title")
      .text(function(d) {return hover_text(d)});

  // Update the nodes ...
  var nodeUpdate = node
      .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });

  nodeUpdate.select("circle")
      .style("r",function(d){
        if (d.is_leaf == 1) {
          return 12.5;
        } else {
          return 10*Math.pow(1.0 + parseFloat(importance[d.name]),2.0); } })
      .style("fill", function(d) { 
        if (d._children) {
          return "#cce6ff"; }
        else if (d.is_leaf == 1) {
          if(calc_probability(d) > 0.5) {
            return "red"
          } else if (calc_probability(d) > 0.1) {
            return "#e6b800"
          } else {
            return "green"
          }
          // return col(calc_probability(d)) }
        } else {
          return "#fff"; }
      })
      .style("stroke","#b3d9ff");
  node.exit().remove();

  // Update the links ...
  var link = svg.selectAll("path.link")
      .data(links, function(d) { return d.target.id; })

  link.enter().insert("path", "g")
      .attr("class", "link")

  link.attr("d", diagonal)
  link.exit().remove()

  // If we choose to see a highlighted patient path:
  if (highlight_flag) {
    // Determine the nodes belonging to the patient path
    patient_Path = return_node_idx(d3.select(nodes)[0][0],Object.assign({}, input));

    // If we chose to see the best recommendation ...
    if (recommend_flag) {
      recs = recommend(root)
      // Save recommendation
//        sessionStorage.recommendation = "Change " + recs["changed"] + " from " + input[recs["changed"]] + " to " + recs[recs["changed"]]; // here
        sessionStorage.recommendation = "Recommendation: Change " + recs["changed"] + " from " + Number(input[recs["changed"]]).toFixed(2) + " to " + Number(recs[recs["changed"]]).toFixed(2);
      recommend_Path = return_node_idx(d3.select(nodes)[0][0],Object.assign({}, recs));
    } else {
      recommend_Path = [];
    }

    // Highlight the user path:
    nodes.forEach(function(d) {
      if (patient_Path.indexOf(d.node_id)>-1) {
        d.highlight = 1;
      }
      if (recommend_Path.indexOf(d.node_id)>-1) {
        d.highlight = 1;
        d.recommend = 1;
      }
    })
  }

  // Update link path sizes:
  link.style("stroke-width",function(d) {return highlight_path(d);})
  
  // Update link colors:
  link.style("stroke",function(d) {
    counter = counter + 1;
    // Change color if it's a patient's biometric path or recommended path:
    if (recommend_flag) {
      if (d.target.recommend == 1 & d.source.recommend == 1 ) { return "#cc5200"; }
      if (d.target.highlight ==1 & d.source.highlight == 1) { return "#000000"; }
    }

    // Default colors are red and green:
    if (counter%2 > 0.5) {
      return "#006600";
    } else {
      return "#800000"; }
  })
  link.style("opacity",0.4)

  // Add name:
  nodeEnter.append("text")
    .attr("x", function(d) { return d.children || d._children ? -16 : 16; })
    .attr("dy", ".35em")
    .attr("text-anchor", function(d) { return d.children || d._children ? "end" : "start"; })
    .text(function(d) {
      if (d.is_leaf == 0) {
        if (d.feature_type == "numeric") {
          if (d.name == "Age") {
            return d.name + " under " + d.threshold.toFixed(0) + " " + d.units + "?";  
          } else {
            return d.name + " under " + d.threshold.toFixed(1) + " " + d.units + "?";
          }
        } else {
          return d.name + "?";
        }
      } else {
        if (d.highlight == 1) {return (calc_probability(d)*100).toFixed(1) + "% chance of HD"}
      }
    })
}

// Toggle children on click.
function click(d) {
  if (d.children) {
    d._children = d.children;
    d.children = null;
  } else {
    d.children = d._children;
    d._children = null;
  }
  update(d);
}

function collapse(d) {
  if (d.children) {
    d._children = d.children;
    d._children.forEach(collapse);
    d.children = null;
  }
}

function hover_text(d) {
  if (d.is_leaf == 1) {
    return (calc_probability(d)*100).toFixed(2) + "% chance of Heart Disease" + "\n" + "Healthy: " + (d.num_negative*(total_healthy/num_healthy)).toFixed() + "\n" + "Diseased: " + d.num_positive;
  } else {
    return d.name + "\n" + "Healthy: " + (d.num_negative*(total_healthy/num_healthy)).toFixed() + "\n" + "Diseased: " + d.num_positive + "\n" + "Importance: " + importance[d.name].toFixed(2);    
  }
}

function calc_probability(d) {
// Function to calculate the printed probability of HD at a leaf node
  // Scale the negative count to a real population (our data has 50/50 heart disease/healthy):
  return (d.num_positive/(d.num_positive+d.num_negative*(total_healthy/num_healthy))).toFixed(3); 
}

function highlight_path(d) {
// Function to highlight the path that our user biometrics would follow:
  if (d.target.recommend == 1) {
    if(d.target.is_leaf == 0 && recommended == false) {
      click(d.target)
    } else if (d.target.is_leaf == 1) {
      recommended = true;
    }
    return 7.5;
  } else if(d.target.highlight == 1) {
    if(d.target.is_leaf == 0 && highlighted == false) {
      click(d.target)
    } else if (d.target.is_leaf == 1) {
      highlighted = true;
    }
    return 7.5;
  } else {
    return 1.5;
  }
}

function return_node_idx(d,biometrics) {
// Starting from the first node check which child satisfies our patient biometric criteria.
// Returns a list of node indices.
  for (i=0;i<d.length;i++) {
    if (d[i].node_id == 0) {
      base_node = d[i];
    }
  }
  // First node is by default picked:
  var patient_nodes = [0];

  // Call initialized recursive subfunction:
  patient_nodes = _return_node_idx(base_node,patient_nodes,biometrics)
  return patient_nodes

  // Recursive subfunction:
  function _return_node_idx(d,patient_nodes,biometrics) {
    if (d.is_leaf == 1) {
      return patient_nodes;
    }
    else {
      var name = d.name;
      var val = Number(d.threshold);

      if (d.feature_type == "categorical") {
        if (biometrics[name] == 1) {kid = 0} else {kid = 1};
        if (d.children) {which_child = d.children[kid]} else {which_child = d._children[kid]}
        patient_nodes.push(which_child.node_id)
        return _return_node_idx(which_child,patient_nodes,biometrics)
      } else {
        if (biometrics[name] <= val) {kid = 0} else {kid = 1};
        if (d.children) {which_child = d.children[kid]} else {which_child = d._children[kid]}
        patient_nodes.push(which_child.node_id)
        return _return_node_idx(which_child,patient_nodes,biometrics)
      }
    }
  }
}

function recommend(d) {
//Function to see what alternative paths can lead to a lower incidence of heart disease:
  var changes = ["Body Weight", "Creatinine", "Calcium",
  "Systolic BP", "Total Cholesterol", "Body Mass Index", "Potassium",
  "Glucose", "Sodium", "LDL Cholesterol", "HDL Cholesterol", "Hemoglobin A1c"];
  var biometric_recommend = Object.assign({}, input),
    best_bio = "",
    best_result = chance_of_hd(d,input),
    best_val = NaN;
    print_recommend = 2; // here
  // Methodically go through and vary each metric to see if we can lower the chances of HD
  for (i=0;i<changes.length;i++) {
    test_bio_low = Object.assign({},input);
    test_bio_high = Object.assign({},input);

    // Vary by 25%:
    test_bio_low[changes[i]] = test_bio_low[changes[i]]*0.75;
    test_bio_high[changes[i]] = test_bio_high[changes[i]]*1.25;

    lower_result = chance_of_hd(d,test_bio_low);
    higher_result = chance_of_hd(d,test_bio_high);
    
      if (lower_result < best_result || higher_result < best_result) {
          print_recommend = 1;
      }

    if (lower_result <= best_result) {
      best_val = test_bio_low[changes[i]];
      best_bio = changes[i]
      best_result = lower_result
    }
    if (higher_result <= best_result) {
      best_val = test_bio_high[changes[i]];
      best_bio = changes[i]
      best_result = higher_result
    }

  }

  if (best_val == NaN) {
    return false;
  } else {
    biometric_recommend[best_bio] = best_val.toString();  //To maintain the same data type
    biometric_recommend["changed"] = best_bio;
    return biometric_recommend
  }
}

function chance_of_hd(d,biometrics) {
// Function to calculate chances of heart disease based on patient biometrics.
  if (d.is_leaf == 1) {
    return calc_probability(d);
  }
  else {
    if (d.feature_type == "categorical") {
      if (biometrics[d.name] == 1) {kid = 0} else {kid = 1};
      if (d.children) {which_child = d.children[kid]} else {which_child = d._children[kid]}
      return chance_of_hd(which_child, biometrics)
    } else {
      if (biometrics[d.name] <= d.threshold) {kid = 0} else {kid = 1};
      if (d.children) {which_child = d.children[kid]} else {which_child = d._children[kid]}
      return chance_of_hd(which_child, biometrics)
    }
  }
}
