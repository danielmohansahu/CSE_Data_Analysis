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

// Population numbers (hardcoded from generated synthea data)
var num_healthy = 3007,
    num_sick = 3009,
    total_healthy = 123000 - num_sick;  // Rough estimate!

// true: highlight user biometric path; false: don't highlight
var highlight_flag = true,
    highlighted = false;    // Used to only update once

var margin = {top: 20, right: 120, bottom: 20, left: 180},
    width = 1400 - margin.right - margin.left,
    height = 700 - margin.top - margin.bottom;

var i = 0,
    counter = 0,
    root;

var col = d3.scale.linear().domain([0,0.5]).range(["green","red"]);

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

//////////////////////////////// UPDATE SYSTEM ////////////////////////////////

d3.json("../tree.json", function(error, flare) {
  if (error) throw error;

  root = flare;
  root.x0 = height / 2;
  root.y0 = 0;

  root.children.forEach(collapse);
  update(root);

  // Call recommendation system:
  recs = recommend(root);
  console.log(recs)

  // Recommendation seems to mess up tree structure; reinitialize tree:
  click(root)
  click(root)
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

  // Add name:
  nodeEnter.append("text")
      .attr("x", function(d) { return d.children || d._children ? -16 : 10; })
      .attr("dy", ".35em")
      .attr("text-anchor", function(d) { return d.children || d._children ? "end" : "start"; })
      .text(function(d) {
        if (d.is_leaf == 0) {
          if (d.feature_type == "numeric") {
            return d.name + " under " + d.threshold.toFixed(1) + " " + d.units + "?";
          } else {
            return d.name + "?";
          }
        }
      })

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
          return col(d.num_positive/(d.num_positive+(d.num_negative*(total_healthy/num_healthy)))) }
        else {
          return "#fff"; }
      })
      .style("stroke","#b3d9ff");
  node.exit().remove();

  // Update the links ...
  var link = svg.selectAll("path.link")
      .data(links, function(d) { return d.target.id; })

  link.enter().insert("path", "g")
      .attr("class", "link")
      .style("stroke",function(d) {
        counter = counter + 1;
        if (counter%2 > 0.5) {
          return "#006600";
        } else {
        return "#800000";
        }
      })
      .style("opacity",0.4)

  link.attr("d", diagonal)
  link.exit().remove()

  if (highlight_flag) {
    // Determine the nodes belonging to the patient path
    patient_Path = return_node_idx(d3.select(nodes)[0][0]);
    // Highlight the user path:
    nodes.forEach(function(d) {
      if (patient_Path.indexOf(d.node_id)>-1) {
        d.highlight = 1
      } else {
        d.highlight = 0
      }
    })
    link.style("stroke-width",function(d) {return highlight_path(d, patient_Path, "size");})
  }
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
    return calc_probability(d) + "\n" + "Healthy: " + (d.num_negative*(total_healthy/num_healthy)).toFixed() + "\n" + "Diseased: " + d.num_positive;
  } else {
    return d.name + "\n" + "Healthy: " + (d.num_negative*(total_healthy/num_healthy)).toFixed() + "\n" + "Diseased: " + d.num_positive + "\n" + "Importance: " + importance[d.name].toFixed(2);    
  }
}

// Function to calculate the printed probability of HD at a leaf node
function calc_probability(d) {
  // Scale the negative count to a real population (our data has 50/50 heart disease/healthy):
  var scaled_negative = d.num_negative*(total_healthy/num_healthy);
  return (d.num_positive/(d.num_positive+scaled_negative)*100).toFixed(1) + "% chance of Heart Disease"; 
}

// Function to highlight the path that our user biometrics would follow:
function highlight_path(d,patient_Path) {
  if(d.target.highlight == 1) {
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

// Starting from the first node check which child satisfies our patient biometric criteria.
function return_node_idx(d) {
  for (i=0;i<d.length;i++) {
    if (d[i].node_id == 0) {
      base_node = d[i];
    }
  }
  // First node is by default picked:
  var patient_nodes = [0];

  // Call initialized recursive subfunction:
  patient_nodes = _return_node_idx(base_node,patient_nodes)
  return patient_nodes

  // Recursive subfunction:
  function _return_node_idx(d,patient_nodes) {
    if (d.is_leaf == 1) {
      return patient_nodes;
    }
    else {
      var name = d.name;
      var val = Number(d.threshold);

      if (d.feature_type == "categorical") {
        if (input[name] == 1) {
          // Return first child
          if (d.children) {
            patient_nodes.push(d.children[0].node_id)
            return _return_node_idx(d.children[0],patient_nodes)
          } else {
            patient_nodes.push(d._children[0].node_id)
            return _return_node_idx(d._children[0],patient_nodes)
          }
        } else {
          // Return second child
          if (d.children) {
            patient_nodes.push(d.children[1].node_id)
            return _return_node_idx(d.children[1],patient_nodes)
          } else {
            patient_nodes.push(d._children[1].node_id)
            return _return_node_idx(d._children[1],patient_nodes)
          }
        }
      } else {
        if (input[name] < val) {
          // Return first child
          if (d.children) {
            patient_nodes.push(d.children[0].node_id)
            return _return_node_idx(d.children[0],patient_nodes)
          } else {
            patient_nodes.push(d._children[0].node_id)
            return _return_node_idx(d._children[0],patient_nodes)
          }
        } else {
          // Return second child
          if (d.children) {
            patient_nodes.push(d.children[1].node_id)
            return _return_node_idx(d.children[1],patient_nodes)
          } else {
            patient_nodes.push(d._children[1].node_id)
            return _return_node_idx(d._children[1],patient_nodes)
          }
        }
      }
    }
  }
}

//Function to see what alternative paths can lead to a lower incidence of heart disease:
function recommend(d) {
  var changes = {"Creatinine":{}, "Body Weight":{}, "Systolic Blood Pressure":{},
  "Total Cholesterol":{}, "Potassium":{}, "Body Mass Index":{} };

  keys = Object.keys(changes);  

  // Methodically go through and vary each metric to see if we can lower the chances of HD
  for (i=0;i<keys.length;i++) {
    changes[keys[i]]["original"] = input[keys[i]];
    changes[keys[i]]["result"] = chance_of_hd(d,input);
    changes[keys[i]]["t1"] = input[keys[i]]*0.9;
    
    var input_test = input;
    input_test[keys[i]] = changes[keys[i]]["t1"];

    changes[keys[i]]["r1"] = chance_of_hd(d,input_test);
  }
  return changes
}

// Function to calculate chances of heart disease based on patient biometrics.
function chance_of_hd(d,biometrics) {
  if (d.is_leaf == 1) {
    return Number(d.num_positive/(d.num_negative+d.num_positive).toFixed());
  }
  else {
    var full_name = d.name.split(" < "),
      name = full_name[0],
      val = Number(full_name[1]);
    
    if (biometrics[name] < val) {kid = 0} else {kid = 1};
    if (d.children) {which_child = d.children[kid]} else {which_child = d._children[kid]}
    return chance_of_hd(which_child, biometrics)
  }
}