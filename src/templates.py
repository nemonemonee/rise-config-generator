robot_rsc = """
<VXC>
    <Structure>
        <Bodies>
            <Body ID="1">
                <Orientation>0,0,0,1</Orientation>
                <OriginPosition>0,0,0</OriginPosition>
                {}
                <MaterialID>
                    {}
                </MaterialID>
                <SegmentID>
                    {}
                </SegmentID>
                <SegmentType>
                    {}
                </SegmentType>
                <ExpansionSignalID>
                    {}
                </ExpansionSignalID>
            </Body>
        </Bodies>
        <Constraints>
            {}
        </Constraints>
    </Structure>
    <Simulator>
        <Signal>
            <ExpansionNum>{}</ExpansionNum>
            <RotationNum>{}</RotationNum>
        </Signal>
        <RecordHistory>
            <RecordStepSize>200</RecordStepSize>
            <RecordVoxel>1</RecordVoxel>
            <RecordLink>0</RecordLink>
            <RecordFixedVoxels>0</RecordFixedVoxels>
        </RecordHistory>
    </Simulator>
    <Save>
        <ResultPath>robot.result</ResultPath>
        <Record>
            <Text>
                <Rescale>0.001</Rescale>
                <Path>robot.history</Path>
            </Text>
        </Record>
    </Save>
</VXC>
"""

shape_template = """
                <X_Voxels>{}</X_Voxels>
                <Y_Voxels>{}</Y_Voxels>
                <Z_Voxels>{}</Z_Voxels>
"""
layer = "<Layer>{}</Layer>"

hinge_constraint = """
            <Constraint>
                <Type>HINGE_JOINT</Type>
                <RigidBodyA>
                    <BodyID>1</BodyID>
                    <SegmentID>{}</SegmentID>
                    <Anchor>{},{},{}</Anchor>
                </RigidBodyA>
                <RigidBodyB>
                    <BodyID>1</BodyID>
                    <SegmentID>{}</SegmentID>
                    <Anchor>{},{},{}</Anchor>
                </RigidBodyB>
                <HingeAAxis>{}, {}, {}</HingeAAxis>
                <HingeBAxis>{}, {}, {}</HingeBAxis>
            </Constraint>

"""

ball_and_socket_constraint = """
            <Constraint>
                <Type>BALL_AND_SOCKET_JOINT</Type>
                <RigidBodyA>
                    <BodyID>1</BodyID>
                    <SegmentID>{}</SegmentID>
                    <Anchor>{},{},{}</Anchor>
                </RigidBodyA>
                <RigidBodyB>
                    <BodyID>1</BodyID>
                    <SegmentID>{}</SegmentID>
                    <Anchor>{},{},{}</Anchor>
                </RigidBodyB>
            </Constraint>
"""

fixed_constraint = """
            <Constraint>
                <Type>FIXED_JOINT</Type>
                <RigidBodyA>
                    <BodyID>1</BodyID>
                    <SegmentID>{}</SegmentID>
                    <Anchor>{},{},{}</Anchor>
                </RigidBodyA>
                <RigidBodyB>
                    <BodyID>1</BodyID>
                    <SegmentID>{}</SegmentID>
                    <Anchor>{},{},{}</Anchor>
                </RigidBodyB>
            </Constraint>
"""

